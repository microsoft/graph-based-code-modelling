using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.MSBuild;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace CSharpDatasetExtractionUtils
{
    /// <summary>
    /// Iterates through the a corpus of C# code and builds all solutions.
    /// </summary>
    public class CSharpCorpusBuilder
    {
        static CSharpCorpusBuilder()
        {
            MSBuildLocator.RegisterDefaults();
        }
        private const string VS_SOLUTION_EXTENSION = ".sln";

        private const int NuGetTimeoutMillis = 30 * 60 * 1000;

        private readonly PersistentResumeList _solutionResumeList;
        private readonly PersistentResumeList _fileResumeList;
        private readonly ConcurrentQueue<string> _workQueue;
        private readonly bool _avoidDuplicateFiles;
        private readonly bool _invokeInParallelPerSyntaxTree;
        private readonly ThreadSafeTextLog _log;

        public event EventHandler<SemanticModel> ObtainedSemanticTree;

        public CSharpCorpusBuilder(string logLocation,
            bool avoidDuplicateFiles = true,
            bool logToStdout = true,
            bool invokeInParallelPerSyntaxTree = true,
            string resumeListDirectory = null)
        {
            _avoidDuplicateFiles = avoidDuplicateFiles;
            _log = new ThreadSafeTextLog(logLocation, alsoWriteToStdout: logToStdout, append: true);
            _invokeInParallelPerSyntaxTree = invokeInParallelPerSyntaxTree;
            _workQueue = new ConcurrentQueue<string>();

            if (resumeListDirectory == null)
            {
                _solutionResumeList = null;
                _fileResumeList = null;
            }
            else
            {
                _solutionResumeList = new PersistentResumeList(Path.Combine(resumeListDirectory, "handled_solutions.txt"));
                _fileResumeList = new PersistentResumeList(Path.Combine(resumeListDirectory, "handled_files.txt"));
            }
        }

        public void AddDuplicates(string corpusRootDir, string duplicatesJsonPath)
        {
            Debug.Assert(_avoidDuplicateFiles);
            var duplicates = JsonConvert.DeserializeObject<List<List<string>>>(File.ReadAllText(duplicatesJsonPath));
            var allDuplicateFiles = duplicates.SelectMany(d => d.Skip(1)).Select(f => Path.Combine(corpusRootDir, f));
            _fileResumeList.BulkAdd(allDuplicateFiles);
        }

        private void SubdirWorker()
        {
            while (_workQueue.TryDequeue(out string subdirectory))
            {
                foreach (var solutionPath in Directory.GetFiles(subdirectory, "*" + VS_SOLUTION_EXTENSION, SearchOption.AllDirectories))
                {
                    if (_solutionResumeList != null && !_solutionResumeList.AddIfNotContained(solutionPath))
                    {
                        continue;
                    }

                    RestorePackages(solutionPath);
                    try
                    {
                        BuildSolution(solutionPath);
                    }
                    catch (AggregateException e)
                    {
                        _log.LogMessage($"Failed to build solution {solutionPath}. Aborting solution. Causes:");
                        foreach (var innerE in e.InnerExceptions)
                        {
                            Console.WriteLine($" {innerE.Message}");
                        }
                    }
                    catch (Exception e)
                    {
                        _log.LogMessage($"Failed to build solution {solutionPath}: {e.Message}.");
                    }
                }
            }
        }

        public void BuildAllSolutionsInDirectory(string directory)
        {
            // This is a bit of a hack reflecting the structure of our crawl.
            // Building overlapping solutions fails, and thus we try to split things by directory:
            foreach (var subdir in Directory.EnumerateDirectories(directory)) {
                _workQueue.Enqueue(subdir);
            }

            // Create a bunch of workers:
            int numberOfWorkers = Environment.ProcessorCount / 4;
            var workers = new Thread[numberOfWorkers];
            for (int i = 0; i < numberOfWorkers; ++i)
            {
                var workerThread = new Thread(new ThreadStart(SubdirWorker));
                workers[i] = workerThread;
                workerThread.Start();
            }

            while (!_workQueue.IsEmpty)
            {
                Thread.Sleep(5000);
            }

            foreach (var worker in workers)
            {
                worker.Join();
            }
        }

        public void BuildSolution(string solutionPath)
        {
            var workspace = MSBuildWorkspace.Create(new Dictionary<string, string> { { "DebugSymbols", "False" } });
            workspace.SkipUnrecognizedProjects = true;
            workspace.WorkspaceFailed +=
                (e, d) => _log.LogMessage($"Error loading project: {d.Diagnostic.Kind}: {d.Diagnostic.Message}");
            Solution solution;
            _log.LogMessage($"Trying to load {solutionPath}.");
            lock (this)
            {
                solution = workspace.OpenSolutionAsync(solutionPath).Result;
            }
            _log.LogMessage($" Loaded {solutionPath}.");

            ProjectDependencyGraph projectGraph = solution.GetProjectDependencyGraph();
            foreach (var projectId in projectGraph.GetTopologicallySortedProjects())
            {
                var projectPath = solution.GetProject(projectId).FilePath;
                Compilation projectCompilation;
                try
                {
                    _log.LogMessage($"Compiling {projectPath}.");
                    projectCompilation = solution.GetProject(projectId).GetCompilationAsync().Result;
                }
                catch (Exception ex)
                {
                    _log.LogMessage($" Exception occured while compiling project {projectPath}. {ex.Message}");
                    continue;
                }
                switch (projectCompilation)
                {
                    case CSharpCompilation cSharpCompilation:
                        if (cSharpCompilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).Any(d => d.GetMessage().Contains("Predefined type")))
                        {
                            _log.LogMessage($"  Manually adding mscorelib to {solution.GetProject(projectId).Name}!");
                            cSharpCompilation = cSharpCompilation
                                .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location));
                        }
                        var firstDiagnostics = cSharpCompilation.GetDiagnostics()
                                                                .Where(d => d.Severity == DiagnosticSeverity.Error)
                                                                .Take(3);
                        foreach (var diagnostic in firstDiagnostics)
                        {
                            _log.LogMessage($"  Compilation issue diagnostic: {diagnostic.GetMessage()}");
                        }
                        _log.LogMessage($" Compilation completed for {solution.GetProject(projectId).Name}, running extraction...");

                        NotifyPerSyntaxTree(cSharpCompilation);
                        break;
                }
            }
        }

        private void NotifyPerSyntaxTree(CSharpCompilation compilation)
        {
            void InvokeForTree(SyntaxTree tree)
            {
                if (!_fileResumeList.AddIfNotContained(tree.FilePath))
                {
                    return;
                }
                var semanticModel = compilation.GetSemanticModel(tree);
                ObtainedSemanticTree.Invoke(this, semanticModel);
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
            _log.LogMessage($"Restoring packages for {projectOrSolution}.");
            var dir = Path.GetDirectoryName(projectOrSolution);

            var cmd = $"restore {Path.GetFileName(projectOrSolution)} -NonInteractive -source https://api.nuget.org/v3/index.json";
            // _log.LogMessage($" In dir {dir} running `nuget {cmd}`");
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
                    _log.LogMessage($" Killing nuget restore for : {projectOrSolution}");
                    proc.Kill();
                }
            }
            catch (Exception e)
            {
                _log.LogMessage($" Exception occurred restoring NuGet: {e.GetType().ToString()}, {e.Message}");
            }

            stopwatch.Stop();
            _log.LogMessage($" Nuget restore took {stopwatch.Elapsed.TotalSeconds} seconds.");
        }
    }
}
