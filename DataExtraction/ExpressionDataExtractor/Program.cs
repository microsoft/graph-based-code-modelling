using System;
using System.IO;
using SourceGraphExtractionUtils;
using Mono.Options;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;

namespace ExpressionDataExtractor
{
    public static class ExtractorOptions
    {
        public static string DataPath;
        public static string GraphOutputPath;
        public static string TypeHierarchyOutputPath;
        public static string DotDebugDirectory;
        public static string ClonePathDir = string.Empty;
        public static bool SequentialComputation = false;
        public static string PhogExtractLocation = null;
        public static bool Debugger = false;

        static readonly internal OptionSet OptionSet =
            new OptionSet
            {
                {
                    "h|?|help",
                    "Print this help.",
                    s => { PrintHelp(Console.Out); Environment.Exit(0); }
                },
                {
                    "clone-ignore-list=",
                    "Path of list of clone files.",
                    filename => ClonePathDir = filename
                },
                {
                    "d|debug",
                    "Launch debugger.",
                    s => Debugger = true
                },
                {
                    "dot-debug=",
                    "Export graphs as dot files into this directory.",
                    s => DotDebugDirectory = s
                },
                {
                    "sequential-computation|s",
                    "Force sequential computation (simplifies debugging).",
                    s => SequentialComputation = true
                },
                {
                    "extract-phog=",
                    "Extract PHOG format data at the given location.",
                    path => PhogExtractLocation = path
                },
            };

        internal static void PrintHelp(TextWriter outWriter)
        {
            Console.WriteLine($"Usage: {typeof(ExpressionDataExtractor).Assembly.GetName().Name} <dataPath> <outputGraphPath> <outputTypeHierPath>");
            OptionSet.WriteOptionDescriptions(outWriter);
        }

        public static void ParseCLIArgs(string[] args)
        {
            var paths = OptionSet.Parse(args);
            if (paths.Count != 3)
            {
                PrintHelp(Console.Error);
                Environment.Exit(1);
            }
            DataPath = paths[0];
            if (!Directory.Exists(DataPath))
            {
                Console.WriteLine($"Data input directory {DataPath} does not exist.");
                Environment.Exit(1);
            }
            GraphOutputPath = paths[1];
            Directory.CreateDirectory(GraphOutputPath);
            Console.WriteLine($"Writing graphs at {GraphOutputPath}");

            TypeHierarchyOutputPath = paths[2];
            Directory.CreateDirectory(TypeHierarchyOutputPath);
            Console.WriteLine($"Writing type hierarchies at {TypeHierarchyOutputPath}");

            if (Debugger)
            {
                System.Diagnostics.Debugger.Launch();
            }
        }
    }

    public class ExpressionDataExtractor
    {
        private static IEnumerable<string> GetFilesToIgnore()
        {
            if (!string.IsNullOrEmpty(ExtractorOptions.ClonePathDir))
            {
                foreach (var line in File.ReadAllLines(ExtractorOptions.ClonePathDir))
                {
                    var parts = line.Split(',');
                    if (parts.Length != 4)
                    {
                        Console.WriteLine($"Could not parse line {line}");
                        continue;
                    }
                    yield return parts[1];
                }
            }
        }

        static void Main(string[] args)
        {
            ExtractorOptions.ParseCLIArgs(args);

            var graphOutputFilenameTemplate = Path.Combine(ExtractorOptions.GraphOutputPath, $"exprs-graph.jsonl");
            var typeHierarchyOutputFile = Path.Combine(ExtractorOptions.TypeHierarchyOutputPath, $"exprs-types.json.gz");
            string extractorLogPath = Path.Combine(ExtractorOptions.GraphOutputPath, "extractor-log");
            string compileLogPath = Path.Combine(ExtractorOptions.GraphOutputPath, "compile-log.txt");

            PhogExtractor phog = null;
            Action<SyntaxNode> additionalExtractors = null;
            if (!string.IsNullOrEmpty(ExtractorOptions.PhogExtractLocation))
            {
                phog = new PhogExtractor(ExtractorOptions.PhogExtractLocation, 7, ExtractorOptions.DataPath);
                additionalExtractors = node => phog.ExtractFor(node);
            }

            using (var extractor = new SourceGraphExtractor(ExtractorOptions.DataPath,
                                            graphOutputFilenameTemplate,
                                            typeHierarchyOutputFile,
                                            ExtractorOptions.DotDebugDirectory,
                                            logPath: extractorLogPath,
                                            additionalExtractors: additionalExtractors))
            {
                
                using (CSharpCorpusBuilder builder = new CSharpCorpusBuilder(compileLogPath,
                    numExtractionWorkers: 20,
                    compilationQueueSize: 10,
                    resumeListPath: Path.Combine(ExtractorOptions.GraphOutputPath, "projectResumeList.txt")))
                {
                    extractor.AddFilesToIgnore(GetFilesToIgnore());
                    builder.SyntaxTreeParsed += extractor.ExtractAllSamplesFromSemanticModel;
                    builder.BuildAllSolutionsInDirectory(ExtractorOptions.DataPath);
                }                
            }

            if (phog != null)
            {
                phog.Dispose();
            }
        }
    }
}
