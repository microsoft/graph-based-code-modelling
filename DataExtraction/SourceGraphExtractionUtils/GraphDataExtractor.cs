using SourceGraphExtractionUtils.Utils;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SourceGraphExtractionUtils
{
    public class SourceGraphExtractor : IDisposable
    {
        private const int MAX_NUM_NODES_PER_GRAPH = 10000;
        private const int ADDITIONAL_NODES_PER_CANDIDATE_SYMBOL = 1000;
        private const int MAX_SYNTAX_BUDGET = 7;
        private const int MAX_DATAFLOW_BUDGET = 2;
        private static readonly int[] OnlySyntaxEdgesCost;
        private static readonly int[] OnlyDataflowEdgesCost;
        private static readonly ISet<string> AllowedSymbolTypeNames;

        static SourceGraphExtractor()
        {
            OnlySyntaxEdgesCost = new int[Enum.GetValues(typeof(SourceGraphEdge)).Length];
            OnlySyntaxEdgesCost[(int)SourceGraphEdge.Child] = 1;
            OnlySyntaxEdgesCost[(int)SourceGraphEdge.NextToken] = 1;
            OnlySyntaxEdgesCost[(int)SourceGraphEdge.LastUsedVariable] = MAX_SYNTAX_BUDGET + MAX_DATAFLOW_BUDGET;

            OnlyDataflowEdgesCost = new int[Enum.GetValues(typeof(SourceGraphEdge)).Length];
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.LastUse] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.LastWrite] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.ComputedFrom] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.LastLexicalUse] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.ReturnsTo] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.FormalArgName] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.GuardedBy] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.GuardedByNegation] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.BindsToSymbol] = 1;
            OnlyDataflowEdgesCost[(int)SourceGraphEdge.LastUsedVariable] = MAX_SYNTAX_BUDGET + MAX_DATAFLOW_BUDGET;

            AllowedSymbolTypeNames = new HashSet<string> {
                "Boolean",
                "Char",
                "SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64",
                "String",
            };
        }

        protected readonly string _typeHierarchyOutputFilename;
        protected readonly string _dotDebugDirectory;

        protected readonly ChunkedJsonGzWriter _writer;
        protected readonly Random _rng = new Random();

        protected readonly string _repositoryRootPath;
        protected readonly PersistentResumeList _fileResumeList;
        protected readonly ThreadSafeTextLog _log;
        protected readonly Action<SyntaxNode> _additionalExtractors;

        public TypeHierarchy TypeHierarchy { get; private set; }

        public SourceGraphExtractor(string repositoryRootPath, string graphOutputFilenameTemplate,
            string typeHierarchyOutputFilename, string dotDebugDir = null, string logPath = null,
            Action<SyntaxNode> additionalExtractors = null)
        {
            if (File.Exists(_typeHierarchyOutputFilename))
            {
                TypeHierarchy = TypeHierarchy.Load(_typeHierarchyOutputFilename);
            }
            else
            {
                TypeHierarchy = new TypeHierarchy();
            }
            _repositoryRootPath = repositoryRootPath;
            _typeHierarchyOutputFilename = typeHierarchyOutputFilename;
            _dotDebugDirectory = dotDebugDir;
            _writer = new ChunkedJsonGzWriter(graphOutputFilenameTemplate, useJsonlFormat: true, resumeIfFilesExist: true);
            _log = new ThreadSafeTextLog(logPath + ".txt");
            _fileResumeList = new PersistentResumeList(logPath + "-fileresumelist.txt");
            _additionalExtractors = additionalExtractors;
        }

        public static bool IsSimpleType(ITypeSymbol typeSymbol)
        {
            if (typeSymbol is IArrayTypeSymbol arrayTypeSymbol)
            {
                return IsSimpleType(arrayTypeSymbol.ElementType);
            }
            else
            {
                return AllowedSymbolTypeNames.Contains(typeSymbol.Name);
            }
        }

        public void AddFilesToIgnore(IEnumerable<string> filepathToIgnore)
        {
            foreach (var file in filepathToIgnore)
            {
                _fileResumeList.AddIfNotContained(file);
            }
        }

        public static bool HasSimpleType(ISymbol symbol)
            => TypeHierarchy.ComputeTypeForSymbol(symbol, out ITypeSymbol symbolType) && IsSimpleType(symbolType);

        #region Output-related methods
        private static void WriteNode(TextWriter writer, SyntaxNodeOrToken node, Dictionary<SyntaxNodeOrToken, int> nodeNumberer)
        {
            if (!nodeNumberer.TryGetValue(node, out int nodeNumber))
            {
                nodeNumber = nodeNumberer.Count;
                nodeNumberer[node] = nodeNumber;
            }

            writer.Write(nodeNumber);
            writer.Write(":");
            if (node.IsNode)
            {
                SyntaxNode syntaxNode = node.AsNode();
                writer.Write(syntaxNode.GetType().Name);
            }
            else
            {
                writer.Write("'");
                writer.Write(node.AsToken().ValueText);
                writer.Write("'");
            }
        }

        private static void WriteGraphAsText(TextWriter writer, SourceGraph graph)
        {
            var nodeNumberer = new Dictionary<SyntaxNodeOrToken, int>();
            foreach (var node in graph.Nodes)
            {
                writer.Write("Edges from ");
                WriteNode(writer, node, nodeNumberer);
                writer.Write(": ");

                bool afterFirst = false;
                foreach (var edge in graph.GetOutEdges(node))
                {
                    if (afterFirst)
                    {
                        writer.Write(", ");
                    }
                    writer.Write("--");
                    writer.Write(edge.Label);
                    writer.Write("-->");
                    WriteNode(writer, edge.Target, nodeNumberer);
                    afterFirst = true;
                }
                writer.WriteLine();
            }
        }

        public static string GetChunkedOutputFilename(string fileName, int chunkNum)
        {
            if (fileName == null)
            {
                return null;
            }
            else if (fileName.EndsWith(".gz"))
            {
                return fileName.Replace(".gz", "." + chunkNum + ".gz");
            }
            else
            {
                return fileName + "." + chunkNum + ".gz";
            }
        }

        public void Dispose()
        {
            _writer.Dispose();
            TypeHierarchy.SaveTypeHierarchy(_typeHierarchyOutputFilename);
        }
        #endregion

        #region Source2Graph (helper) methods
        /// <summary>
        /// Compute a mapping from nodes that we want to remove to syntax tokens that serve as replacements.
        /// </summary>
        /// <param name="graph"></param>
        /// <returns></returns>
        private static IDictionary<SyntaxNode, SyntaxNodeOrToken> ComputeNodeCompressionMap(SourceGraph graph)
        {
            var nodeToTokenMap = new Dictionary<SyntaxNode, SyntaxNodeOrToken>();

            foreach (var node in graph.Nodes.Where(node => node.IsNode))
            {
                var syntaxNode = node.AsNode();
                SyntaxNodeOrToken? replacementNode = null;
                switch (syntaxNode)
                {
                    case PropertyDeclarationSyntax propDecl:
                        replacementNode = propDecl.Identifier;
                        break;
                    case VariableDeclaratorSyntax varDecl:
                        replacementNode = varDecl.Identifier;
                        break;
                    case IdentifierNameSyntax idNameSyn:
                        replacementNode = idNameSyn.Identifier;
                        break;
                    case OmittedArraySizeExpressionSyntax ommittedSyn:
                        replacementNode = ommittedSyn.OmittedArraySizeExpressionToken;
                        break;
                    case PredefinedTypeSyntax predefTypeSyn:
                        replacementNode = predefTypeSyn.Keyword;
                        break;
                    case LiteralExpressionSyntax litSyn:
                        replacementNode = litSyn.Token;
                        break;
                    case EnumMemberDeclarationSyntax enumMemberSym:
                        replacementNode = enumMemberSym.Identifier;
                        break;
                    case SimpleBaseTypeSyntax simpleTypeSyntax:
                        replacementNode = simpleTypeSyntax.Type;
                        break;
                    case TypeParameterSyntax typeParSyntax:
                        replacementNode = typeParSyntax.Identifier;
                        break;
                    case BaseExpressionSyntax baseExprSyntax:
                        replacementNode = baseExprSyntax.Token;
                        break;
                    case SingleVariableDesignationSyntax singleVarSyntax:
                        replacementNode = singleVarSyntax.Identifier;
                        break;
                    case ThisExpressionSyntax thisSyntax:
                        replacementNode = thisSyntax.Token;
                        break;
                    case ClassOrStructConstraintSyntax classOrStructSyntax:
                        replacementNode = classOrStructSyntax.ClassOrStructKeyword;
                        break;
                    case InterpolatedStringTextSyntax interpolStringSyntax:
                        replacementNode = interpolStringSyntax.TextToken;
                        break;
                    case ArgumentSyntax argSyn:
                        if (argSyn.NameColon == null && argSyn.RefOrOutKeyword.SyntaxTree == null)
                        {
                            replacementNode = argSyn.Expression;
                        }
                        break;
                }

                if (replacementNode.HasValue)
                {
                    nodeToTokenMap[syntaxNode] = replacementNode.Value;
                }
            }

            return nodeToTokenMap;
        }

        private static SourceGraph GetCompressedGraph(SourceGraph graph)
        {
            var compressedGraph = SourceGraph.Create(graph);
            var nodeToReplacement = ComputeNodeCompressionMap(graph);

            foreach (var sourceNode in graph.Nodes)
            {
                var newSourceNode = sourceNode;
                if (sourceNode.IsNode)
                {
                    if (nodeToReplacement.TryGetValue(sourceNode.AsNode(), out var replacementSourceNode))
                    {
                        newSourceNode = replacementSourceNode;
                    }
                }

                foreach (var edge in graph.GetOutEdges(sourceNode))
                {
                    SyntaxNodeOrToken newTargetNode = edge.Target;
                    if (edge.Target.IsNode)
                    {
                        var targetSyntaxNode = edge.Target.AsNode();
                        while (targetSyntaxNode != null && nodeToReplacement.TryGetValue(targetSyntaxNode, out SyntaxNodeOrToken replacementTargetNode))
                        {
                            if (replacementTargetNode.IsNode)
                            {
                                targetSyntaxNode = replacementTargetNode.AsNode();
                            }
                            else
                            {
                                targetSyntaxNode = null;
                            }
                            newTargetNode = replacementTargetNode;
                        }
                    }

                    //Don't make links between replaced nodes into cycles:
                    if (newSourceNode == newTargetNode && sourceNode != edge.Target)
                    {
                        continue;
                    }

                    compressedGraph.AddEdge(newSourceNode, edge.Label, newTargetNode);
                }
            }

            return compressedGraph;
        }

        private static void AddNextLexicalUseEdges(SourceGraph graph)
        {
            var allTokens = graph.SemanticModel.SyntaxTree.GetRoot().DescendantTokens().Where(t => t.Text.Length > 0).ToArray();

            var nameToLastUse = new Dictionary<string, SyntaxToken>();
            for (int i = 0; i < allTokens.Length; ++i)
            {
                var curTok = allTokens[i];
                if (graph.UsedVariableNodes.Contains(curTok))
                {
                    var curName = curTok.Text;
                    if (nameToLastUse.TryGetValue(curName, out var lastUseTok))
                    {
                        graph.AddEdge(curTok, SourceGraphEdge.LastLexicalUse, lastUseTok);
                    }
                    nameToLastUse[curName] = curTok;
                }
            }
        }

        public static SourceGraph ExtractSourceGraph(SemanticModel semanticModel, SyntaxToken[] allTokens)
        {
            var tree = semanticModel.SyntaxTree;
            var graph = ASTGraphExtractor.ConstructASTGraph(semanticModel);
            MethodInfoGraphExtractor.ConstructMethodInfoGraph(semanticModel, graph);
            VariableUseGraphExtractor.ConstructVariableUseGraph(semanticModel, graph);
            DataFlowGraphHelper.AddDataFlowEdges(graph, graph.UsedVariableNodes);
            try
            {
                GuardAnnotationGraphExtractor.AddGuardAnnotationsToGraph(graph, semanticModel.SyntaxTree.GetRoot());
            }
            catch (Exception e)
            {
                Console.WriteLine("Failed to add guardian annotations. Error message: " + e.Message);
            }

            AddNextLexicalUseEdges(graph);
            var compressedGraph = GetCompressedGraph(graph);
            return compressedGraph;
        }
        #endregion

        private static void GetReachableNodes(SourceGraph graph,
            int[] edgeTypeToCost,
            int startCostBudget,
            SyntaxNodeOrToken startNode,
            ICollection<SyntaxNodeOrToken> allowedNodes,
            Dictionary<SyntaxNodeOrToken, int> result,
            Predicate<Edge<SyntaxNodeOrToken, SourceGraphEdge>> followEdge)
        {
            var todo = new Stack<(SyntaxNodeOrToken node, int budgetLimit)>();
            todo.Push((node: startNode, budgetLimit: startCostBudget));

            while (todo.Count > 0)
            {
                var (node, budgetLimit) = todo.Pop();
                var (inEdges, outEdges) = graph.GetEdges(node);

                foreach (var edge in inEdges.Where(followEdge.Invoke))
                {
                    var partnerNode = edge.Source;
                    var edgeCost = edgeTypeToCost[(int)edge.Label];
                    var newNodeBudget = budgetLimit - edgeCost;
                    if (newNodeBudget > 0 && allowedNodes.Contains(partnerNode))
                    {
                        //Check if we've already reached this with lower cost -- if yes, we can skip the visit.
                        if (result.TryGetValue(partnerNode, out var oldNodeBudget) && oldNodeBudget >= newNodeBudget)
                        {
                            continue;
                        }

                        result[partnerNode] = newNodeBudget;
                        todo.Push((node: partnerNode, budgetLimit: newNodeBudget));
                    }
                }

                foreach (var edge in outEdges.Where(followEdge.Invoke))
                {
                    var partnerNode = edge.Target;
                    var edgeCost = edgeTypeToCost[(int)edge.Label];
                    var newNodeBudget = budgetLimit - edgeCost;
                    if (newNodeBudget > 0 && allowedNodes.Contains(partnerNode))
                    {
                        //Check if we've already reached this with lower cost -- if yes, we can skip the visit.
                        if (result.TryGetValue(partnerNode, out var oldNodeBudget) && oldNodeBudget >= newNodeBudget)
                        {
                            continue;
                        }

                        result[partnerNode] = newNodeBudget;
                        todo.Push((node: partnerNode, budgetLimit: newNodeBudget));
                    }
                }
            }
        }

        protected static void CopySubgraphAroundHole(
            SourceGraph sourceGraph,
            IEnumerable<SyntaxNodeOrToken> targetNodes,
            ISet<SyntaxNodeOrToken> forbiddenNodes,
            SourceGraph resultGraph,
            int numSymbolsAddedSoFar = 0)
        {
            var allowedNodes = new HashSet<SyntaxNodeOrToken>(sourceGraph.Nodes.Where(n => !forbiddenNodes.Contains(n)));

            /// <summary>
            /// Filter edges that are "active" in the output graph.
            ///
            /// The current implementation allows all edges except from the GuardedBy[Negation].
            /// GuardedBy[Negation] edges are inlcuded only if the label of the source node
            /// is in the set of identifiers of the descendants of the target node.
            /// </summary>
            bool IsActiveEdge(Edge<SyntaxNodeOrToken, SourceGraphEdge> edge)
            {
                if (edge.Label != SourceGraphEdge.GuardedBy && edge.Label != SourceGraphEdge.GuardedByNegation)
                {
                    return true;
                }

                var sourceNodeLabel = edge.Source.ToString();
                if (edge.Target.IsToken)
                {
                    return edge.Target.ToString().Equals(sourceNodeLabel);
                }

                // If our target token (the only element of forbidden nodes at this point) is a descendant of the target,
                // we are considering a slot inside a condition. In that case, keep all GuardedBy edges.
                if (edge.Target.AsNode().DescendantTokens().Any(token => forbiddenNodes.Contains(token)))
                {
                    return true;
                }

                return edge.Target.AsNode().DescendantTokens()
                                           .Where(t => allowedNodes.Contains(t))
                                           .Any(t => t.ToString().Equals(sourceNodeLabel));
            }

            var nodesToSyntaxRemainingBudget = new Dictionary<SyntaxNodeOrToken, int>();
            foreach (var targetNode in targetNodes)
            {
                GetReachableNodes(sourceGraph, OnlySyntaxEdgesCost, MAX_SYNTAX_BUDGET, targetNode, allowedNodes, nodesToSyntaxRemainingBudget, IsActiveEdge);
            }

            var nodesToDataflowRemainingBudget = new Dictionary<SyntaxNodeOrToken, int>();
            foreach (var targetNode in targetNodes)
            {
                GetReachableNodes(sourceGraph, OnlyDataflowEdgesCost, MAX_DATAFLOW_BUDGET, targetNode, nodesToSyntaxRemainingBudget.Keys, nodesToDataflowRemainingBudget, IsActiveEdge);
                // This ensures that all targetNodes appear in the nodesToDataflowRemainingBudget
                nodesToDataflowRemainingBudget[targetNode] = MAX_DATAFLOW_BUDGET + MAX_SYNTAX_BUDGET;
            }

            // Order the nodes by cost so that nodes smaller cost are reached first.
            foreach (var node in nodesToDataflowRemainingBudget.Keys.OrderByDescending(n => nodesToDataflowRemainingBudget[n]))
            {
                foreach (var edge in sourceGraph.GetOutEdges(node))
                {
                    if (edge.Label == SourceGraphEdge.LastUsedVariable) continue;
                    if (resultGraph.CountNodes > MAX_NUM_NODES_PER_GRAPH + ADDITIONAL_NODES_PER_CANDIDATE_SYMBOL * numSymbolsAddedSoFar &&
                        !(resultGraph.ContainsNode(node) && resultGraph.ContainsNode(edge.Target))) continue;
                    if (nodesToDataflowRemainingBudget.ContainsKey(edge.Target) && !forbiddenNodes.Contains(node) && !forbiddenNodes.Contains(edge.Target))
                    {
                        resultGraph.AddEdge(node, edge.Label, edge.Target);
                    }
                }
            }
        }
    }
}
