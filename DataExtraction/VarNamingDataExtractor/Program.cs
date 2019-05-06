using System;
using System.IO;
using SourceGraphExtractionUtils;
using Mono.Options;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using CSharpDatasetExtractionUtils;

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

            using (var extractor = new VarNamingGraphExtractor(ExtractorOptions.DataPath,
                                            graphOutputFilenameTemplate,
                                            typeHierarchyOutputFile,
                                            ExtractorOptions.DotDebugDirectory,
                                            logPath: extractorLogPath))
            {
                var builder = new CSharpCorpusBuilder(
                    logLocation: Path.Combine(ExtractorOptions.GraphOutputPath, "log", "compile-log.txt"),
                    resumeListDirectory: Path.Combine(ExtractorOptions.GraphOutputPath, "log"));
                {
                    extractor.AddFilesToIgnore(GetFilesToIgnore());
                    builder.ObtainedSemanticTree += extractor.ExtractAllSamplesFromSemanticModel;
                    builder.BuildAllSolutionsInDirectory(ExtractorOptions.DataPath);
                }
            }
        }
    }
}
