using SourceGraphExtractionUtils;
using SourceGraphExtractionUtils.Utils;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.MSBuild;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ExpressionDataExtractor
{
    /// <summary>
    /// Iterates through the a corpus of C# code and builds all solutions.
    /// </summary>
    public class CSharpCorpusBuilder : IDisposable
    {
        static CSharpCorpusBuilder()
        {
            MSBuildLocator.RegisterDefaults();
        }

        private const string VS_SOLUTION_EXTENSION = ".sln";
        private const string CSHARP_PRJ_SOLUTION_EXTENSION = ".csproj";

        private const int NuGetTimeoutMillis = 30 * 60 * 1000;

        private readonly PersistentResumeList _projectResumeList;

        public CSharpCorpusBuilder(string errorLogLocation, bool avoidDuplicateFiles = true,
            bool logToStdout = true, bool invokeInParallelPerSyntaxTree = true,
            int numExtractionWorkers = 1, int compilationQueueSize = 1, string resumeListPath = null)
        {
            _avoidDuplicateFiles = avoidDuplicateFiles;
            _log = new ThreadSafeTextLog(errorLogLocation, alsoWriteToStdout: logToStdout, append: true);
            _invokeInParallelPerSyntaxTree = invokeInParallelPerSyntaxTree;
            _compilationQueue = new BlockingCollection<CSharpCompilation>(
                new ConcurrentQueue<CSharpCompilation>(), compilationQueueSize);

            _threadWorkers = new List<Task>();
            for (int i = 0; i < numExtractionWorkers; i++)
            {
                _threadWorkers.Add(Task.Run(() => ThreadWorker()));
            }
            if (resumeListPath == null)
            {
                _projectResumeList = null;
            }
            else
            {
                _projectResumeList = new PersistentResumeList(resumeListPath);
            }
        }

        private readonly bool _avoidDuplicateFiles;
        private readonly bool _invokeInParallelPerSyntaxTree;
        private readonly List<Task> _threadWorkers;
        private readonly ConcurrentDictionary<string, bool> _seenFiles = new ConcurrentDictionary<string, bool>();
        private readonly ConcurrentDictionary<string, bool> _seenProjects = new ConcurrentDictionary<string, bool>();
        private readonly ThreadSafeTextLog _log;
        private readonly BlockingCollection<CSharpCompilation> _compilationQueue;

        public event EventHandler<CSharpCompilation> CompilationCompleted;
        public event EventHandler<SemanticModel> SyntaxTreeParsed;

        public void BuildAllSolutionsInDirectory(string directory)
        {
            _log.LogMessage($"Starting building all solutions in {directory}");
            foreach (var solution in Directory.EnumerateFiles(directory, "*" + VS_SOLUTION_EXTENSION, SearchOption.AllDirectories))
            {
                try
                {                    
                    BuildSolution(solution);                    
                }
                catch (Exception e)
                {
                    _log.LogMessage($"Failed on extraction: {e.Message}. {e.StackTrace}");
                }
            }
        }

        public void BuildSolution(string solutionPath)
        {
            Debug.Assert(solutionPath.EndsWith(VS_SOLUTION_EXTENSION));
            RestorePackages(solutionPath);

            MSBuildWorkspace workspace = CreateWorkspace();
            Solution solution = workspace.OpenSolutionAsync(solutionPath).Result;

            BuildSolution(solution);
        }

        public void BuildAllProjectsPerFolder(string directory)
        {
            foreach (var folder in Directory.EnumerateDirectories(directory))
            {
                var folderToSearch = Path.Combine(directory, folder);
                BuildCsProjInFolder(folderToSearch);
            }
        }

        public void BuildCsProjInFolder(string folderToSearch)
        {
            IEnumerable<string> projectFiles =
                                (new DirectoryInfo(folderToSearch))
                                .EnumerateFiles("*" + CSHARP_PRJ_SOLUTION_EXTENSION, SearchOption.AllDirectories)
                                .Select(fileInfo => Path.Combine(folderToSearch, fileInfo.FullName));

            MSBuildWorkspace workspace = CreateWorkspace();
            foreach (string projectFile in projectFiles)
            {
                RestorePackages(projectFile);
                bool alreadyAdded = workspace.CurrentSolution.Projects.Any(p => p.FilePath.Equals(projectFile, StringComparison.OrdinalIgnoreCase));
                if (!alreadyAdded)
                {
                    var project = workspace.OpenProjectAsync(projectFile).Result;
                }
            }
            BuildSolution(workspace.CurrentSolution);
        }

        private MSBuildWorkspace CreateWorkspace()
        {
            var workspace = MSBuildWorkspace.Create(new Dictionary<string, string> { { "DebugSymbols", "False" } });
            workspace.SkipUnrecognizedProjects = true;
            workspace.WorkspaceFailed +=
                (e, d) => _log.LogMessage($"Error loading project: {d.Diagnostic.Kind}: {d.Diagnostic.Message}");
            return workspace;
        }

        public void BuildSolution(Solution solution)
        {
            _log.LogMessage($"Starting build of {solution.FilePath}");
            ProjectDependencyGraph projectGraph = solution.GetProjectDependencyGraph();
            foreach (var projectId in projectGraph.GetTopologicallySortedProjects())
            {
                if (!_seenProjects.TryAdd(solution.GetProject(projectId).FilePath, true))
                {
                    continue;
                }
                if (_projectResumeList != null && !_projectResumeList.AddIfNotContained(solution.GetProject(projectId).FilePath))
                {
                    _log.LogMessage($"Skipping {solution.GetProject(projectId).FilePath}, as it was processed in an earlier run.");
                    continue;
                }

                Compilation projectCompilation;
                try
                {
                    projectCompilation = solution.GetProject(projectId).GetCompilationAsync().Result;
                }
                catch (Exception ex)
                {
                    _log.LogMessage($"Exception occured while compiling project {projectId}. {ex.Message}: {ex.StackTrace}");
                    continue;
                }
                switch (projectCompilation)
                {
                    case CSharpCompilation cSharpCompilation:
                        if (cSharpCompilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).Any(d => d.GetMessage().Contains("Predefined type")))
                        {
                            _log.LogMessage($"Manually adding mscorelib to {solution.GetProject(projectId).Name}!");
                            cSharpCompilation = cSharpCompilation
                                .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location));
                        }
                        foreach (var diagnostic in cSharpCompilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error))
                        {
                            _log.LogMessage($"Compilation issue diagnostic: {diagnostic.GetMessage()}");
                        }
                        _log.LogMessage($"Compilation completed for {solution.GetProject(projectId).Name}, running extraction...");

                        _compilationQueue.Add(cSharpCompilation);
                        break;
                }
            }
        }

        private void ThreadWorker()
        {
            while (!_compilationQueue.IsCompleted)
            {
                var taken = _compilationQueue.TryTake(out var nextCompilation, 1000);
                if (!taken) continue;
                CompilationCompleted?.Invoke(this, nextCompilation);
                NotifyPerSyntaxTree(nextCompilation);
            }
        }

        private void NotifyPerSyntaxTree(CSharpCompilation compilation)
        {
            void InvokeForTree(SyntaxTree tree)
            {
                if (_avoidDuplicateFiles && !_seenFiles.TryAdd(tree.FilePath, true))
                {
                    return;
                }
                var semanticModel = compilation.GetSemanticModel(tree);
                SyntaxTreeParsed?.Invoke(this, semanticModel);
            }

            if (_invokeInParallelPerSyntaxTree)
            {
                Parallel.ForEach(compilation.SyntaxTrees, InvokeForTree);
            }
            else
            {
                foreach (var tree in compilation.SyntaxTrees)
                {
                    InvokeForTree(tree);
                }
            }
        }

        private void RestorePackages(string projectOrSolution)
        {
            _log.LogMessage($"Restoring packages for: {projectOrSolution}");
            var dir = Path.GetDirectoryName(projectOrSolution);

            var cmd = $"restore {Path.GetFileName(projectOrSolution)} -NonInteractive -source https://api.nuget.org/v3/index.json";
            _log.LogMessage($"In dir {dir} running nuget {cmd}");
            var startInfo = new ProcessStartInfo("nuget", cmd)
            {
                WorkingDirectory = dir,
                RedirectStandardOutput = false,
                WindowStyle = ProcessWindowStyle.Hidden,
                UseShellExecute = true
            };

            var stopwatch = new Stopwatch();
            try
            {
                var proc = Process.Start(startInfo);
                proc.WaitForExit(NuGetTimeoutMillis);
                if (!proc.HasExited)
                {
                    _log.LogMessage($"Killing nuget restore for : {projectOrSolution}");
                    proc.Kill();
                }
            }
            catch (Exception e)
            {
                _log.LogMessage($"Exception occurred restoring NuGet: {e.GetType().ToString()}, {e.Message}");
            }

            stopwatch.Stop();
            _log.LogMessage($"Nuget restore took {stopwatch.Elapsed.TotalMinutes} minutes.");
        }

        public void WaitAllFinished()
        {
            Console.WriteLine("Compilations completed, completing extraction jobs...");
            _compilationQueue.CompleteAdding();
            foreach (var task in _threadWorkers)
            {
                task.Wait();
            }
        }

        public void Dispose()
        {
            WaitAllFinished();
        }
    }
}
