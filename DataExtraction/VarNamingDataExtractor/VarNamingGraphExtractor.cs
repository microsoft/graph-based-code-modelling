using DataExtraction.Utils;
using SourceGraphExtractionUtils.Utils;
using Microsoft.CodeAnalysis;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SourceGraphExtractionUtils
{
    public class VarNamingGraphExtractor : SourceGraphExtractor
    {
        public class NamingGraphInformation
        {
            public readonly SourceGraph ContextGraph;
            public readonly IList<SyntaxToken> SlotTokens;
            private readonly string _repositoryRootPath;

            public NamingGraphInformation(string repositoryRootPath, SourceGraph contextGraph, IList<SyntaxToken> slotTokens)
            {
                _repositoryRootPath = repositoryRootPath;
                ContextGraph = contextGraph;
                SlotTokens = slotTokens;
            }

            public (Dictionary<SyntaxNodeOrToken, int> NodeNumberer, Dictionary<SyntaxNodeOrToken, string> NodeLabeler) WriteJson(JsonWriter jWriter)
            {
                jWriter.WriteStartObject();
                jWriter.WritePropertyName("filename");
                if (ContextGraph.SemanticModel.SyntaxTree.FilePath.StartsWith(_repositoryRootPath))
                {
                    jWriter.WriteValue(ContextGraph.SemanticModel.SyntaxTree.FilePath.Substring(_repositoryRootPath.Length));
                }
                else
                {
                    jWriter.WriteValue(ContextGraph.SemanticModel.SyntaxTree.FilePath);
                }

                var nodeNumberer = new Dictionary<SyntaxNodeOrToken, int>();
                var dummyNodeLabeler = new Dictionary<SyntaxNodeOrToken, string>();
                int i = 0;
                foreach (var slotToken in SlotTokens)
                {
                    dummyNodeLabeler[slotToken] = "<SLOT>";
                    nodeNumberer[slotToken] = i++;
                }

                jWriter.WritePropertyName("ContextGraph");
                ContextGraph.WriteJson(jWriter, nodeNumberer, dummyNodeLabeler);

                jWriter.WritePropertyName("TargetName");
                jWriter.WriteValue(SlotTokens.First().Text);

                jWriter.WritePropertyName("SlotNodes");
                jWriter.WriteStartArray();
                foreach (var slotToken in SlotTokens)
                {
                    jWriter.WriteValue(nodeNumberer[slotToken]);
                }
                jWriter.WriteEndArray();

                jWriter.WriteEndObject();

                return (NodeNumberer: nodeNumberer, NodeLabeler: dummyNodeLabeler);
            }
        }

        public VarNamingGraphExtractor(string repositoryRootPath, string graphOutputFilenameTemplate,
            string typeHierarchyOutputFilename, string dotDebugDir = null, string logPath = null,
            Action<SyntaxNode> additionalExtractors = null)
            : base(repositoryRootPath, graphOutputFilenameTemplate, typeHierarchyOutputFilename, dotDebugDir, logPath, additionalExtractors)
        { }

        public static Dictionary<ISymbol, List<int>> BuildSymbolToUsagesMap(
                SemanticModel semanticModel,
                SyntaxToken[] allTokens,
                ISet<ISymbol> variableSymbolsToConsider)
        {
            var symbolToTokens = new Dictionary<ISymbol, List<int>>();

            for (var i = 0; i < allTokens.Length; i++)
            {
                var token = allTokens[i];
                var symbol = RoslynUtils.GetTokenSymbolReference(token, semanticModel);
                if (symbol == null) continue;

                symbol = symbol.OriginalDefinition;
                if (!variableSymbolsToConsider.Contains(symbol)) continue;

                if (!symbolToTokens.TryGetValue(symbol, out var tokenListForSymbol))
                {
                    tokenListForSymbol = new List<int>();
                    symbolToTokens[symbol] = tokenListForSymbol;
                }

                tokenListForSymbol.Add(i);
            }

            return symbolToTokens;
        }

        public void ExtractAllSamplesFromSemanticModel(object sender, SemanticModel semanticModel)
        {
            var tree = semanticModel.SyntaxTree;
            if (_fileResumeList.AddIfNotContained(tree.FilePath))
            {
                int numVariablesExtracted = 0;
                try
                {
                    if (tree.FilePath.Contains("TemporaryGeneratedFile") || tree.FilePath.EndsWith(".Generated.cs"))
                    {
                        Console.WriteLine($"Ignoring file {tree.FilePath}.");
                        return; // TODO: Remove any temp files in obj
                    }
                    //Console.WriteLine($"Writing out samples for {tree.FilePath}");

                    var addedTypes = new HashSet<ITypeSymbol>();
                    foreach (var syntaxNode in tree.GetRoot().DescendantNodes())
                    {
                        var symbol = RoslynUtils.GetReferenceSymbol(syntaxNode, semanticModel);
                        if (RoslynUtils.GetTypeSymbol(symbol, out var typeSymbol) && addedTypes.Add(typeSymbol))
                        {
                            TypeHierarchy.Add(typeSymbol);
                        }
                    }

                    // Compute the list of non-empty tokens.
                    var allTokens = tree.GetRoot().DescendantTokens().Where(t => t.Text.Length > 0).ToArray();

                    // Compute the used variables that we are interested in:
                    var variableSymbolCandidates = new HashSet<ISymbol>(RoslynUtils.GetUsedVariableSymbols(semanticModel, tree.GetRoot()));
                    var symbolToUsageTokens = BuildSymbolToUsagesMap(semanticModel, allTokens, variableSymbolCandidates);
                    var variableSymbolsToUse = RoslynUtils.ComputeVariablesToConsider(variableSymbolCandidates, symbolToUsageTokens.Keys);

                    SourceGraph sourceGraph = null;
                    foreach (var symbol in variableSymbolsToUse)
                    {
                        if (symbolToUsageTokens[symbol].Count <= 1) continue;
                        var variableTokens = symbolToUsageTokens[symbol].Select(i => allTokens[i]).ToList();
                        if (sourceGraph == null)
                        {
                            try
                            {
                                sourceGraph = ExtractSourceGraph(semanticModel, allTokens); // Compute the graph if we haven't done so yet
                            }
                            catch (Exception e)
                            {
                                Console.WriteLine($"Error while extracting graph. Aborting file. Cause: {e.Message}");
                                return;
                            }
                        }

                        var resultGraph = SourceGraph.Create(sourceGraph);
                        // Limit around variable tokens
                        CopySubgraphAroundHole(
                            sourceGraph,
                            new List<SyntaxNodeOrToken>(variableTokens.Select(t =>
                            {
                                SyntaxNodeOrToken snt = t;
                                return snt;
                            })),
                            new HashSet<SyntaxNodeOrToken>(),
                            resultGraph);

                        ++numVariablesExtracted;
                        var sample = new NamingGraphInformation(_repositoryRootPath, resultGraph, variableTokens);
                        _writer.WriteElement(jw => sample.WriteJson(jw));
                    }
                    if (numVariablesExtracted > 0)
                    {
                        _log.LogMessage($"Extracted {numVariablesExtracted} var naming samples from {tree.FilePath}.");
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Exception when extracting data from file: {e.Message}: {e.StackTrace}");
                    _log.LogMessage($"Error{e.Message}: {e.StackTrace} while extracting. Managed to extract {numVariablesExtracted} var naming samples from {tree.FilePath}.");
                }
                if (_rng.NextDouble() > .9)
                {
                    // Once in a while save the type hierarchy
                    TypeHierarchy.SaveTypeHierarchy(_typeHierarchyOutputFilename);
                }
            }
        }
    }
}
