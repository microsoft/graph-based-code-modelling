using DataExtraction.Utils;
using SourceGraphExtractionUtils.Utils;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Text;

namespace SourceGraphExtractionUtils
{
    public enum ExpansionSymbolKind { Expression, Token, IntLiteral, CharLiteral, StringLiteral, Variable }

    public class ProducedSymbol
    {
        public readonly uint Id;
        public readonly ExpansionSymbolKind SymbolKind;
        public readonly object Label;

        public ProducedSymbol(uint id, ExpansionSymbolKind symbolKind, object label)
        {
            Id = id;
            SymbolKind = symbolKind;
            Label = label;
        }
    }

    public class HoleContextInformation
    {
        private const int NUM_TOKENS_CONTEXT = 10;

        private readonly string _repositoryRootPath;
        public readonly SourceGraph ContextGraph;
        public readonly SyntaxNode DeletedNode;
        public readonly SyntaxNodeOrToken HoleNode;
        public readonly SyntaxToken LastTokenBeforeHole;
        public readonly List<SyntaxToken> LastUseOfVariablesInScope;
        public readonly IEnumerable<(ProducedSymbol nonTerminal, IEnumerable<ProducedSymbol> production)> Productions;

        public HoleContextInformation(string repositoryRootPath, SyntaxNode deletedNode, SourceGraph contextGraph,
            SyntaxNodeOrToken holeNode, SyntaxToken lastTokenBeforeHole, List<SyntaxToken> lastUseOfVariablesInScope,
            IEnumerable<(ProducedSymbol nonTerminal, IEnumerable<ProducedSymbol> production)> productions)
        {
            _repositoryRootPath = repositoryRootPath;
            ContextGraph = contextGraph;
            DeletedNode = deletedNode;
            HoleNode = holeNode;
            LastTokenBeforeHole = lastTokenBeforeHole;
            LastUseOfVariablesInScope = lastUseOfVariablesInScope;
            Productions = productions;
        }

        public (Dictionary<SyntaxNodeOrToken, int> NodeNumberer, Dictionary<SyntaxNodeOrToken, string> NodeLabeler) WriteJson(JsonWriter jWriter)
        {
            jWriter.WriteStartObject();
            (Dictionary<SyntaxNodeOrToken, int> nodeNumberer, Dictionary<SyntaxNodeOrToken, string> dummyNodeLabeler) = WriteJsonProperties(jWriter);
            jWriter.WriteEndObject();

            return (NodeNumberer: nodeNumberer, NodeLabeler: dummyNodeLabeler);
        }

        private int GetNodeNumber(Dictionary<SyntaxNodeOrToken, int> nodeNumberer, SyntaxNodeOrToken node)
        {
            if (!nodeNumberer.TryGetValue(node, out var nodeNumber))
            {
                nodeNumber = nodeNumberer.Count;
                nodeNumberer[node] = nodeNumber;
            }
            return nodeNumber;
        }

        public (Dictionary<SyntaxNodeOrToken, int> NodeNumberer, Dictionary<SyntaxNodeOrToken, string> NodeLabeler) WriteJsonProperties(JsonWriter jWriter)
        {
            jWriter.WritePropertyName("Filename");
            if (ContextGraph.SemanticModel.SyntaxTree.FilePath.StartsWith(_repositoryRootPath))
            {
                jWriter.WriteValue(ContextGraph.SemanticModel.SyntaxTree.FilePath.Substring(_repositoryRootPath.Length));
            }
            else
            {
                jWriter.WriteValue(ContextGraph.SemanticModel.SyntaxTree.FilePath);
            }
            jWriter.WritePropertyName("HoleSpan");
            jWriter.WriteValue(DeletedNode.FullSpan.ToString());

            jWriter.WritePropertyName("HoleLineSpan");
            jWriter.WriteValue(DeletedNode.SyntaxTree.GetLineSpan(DeletedNode.Span).Span.ToString());

            jWriter.WritePropertyName("OriginalExpression");
            jWriter.WriteValue(DeletedNode.ToString());

            var nodeNumberer = new Dictionary<SyntaxNodeOrToken, int>();
            var dummyNodeLabeler = new Dictionary<SyntaxNodeOrToken, string> { { HoleNode, "<HOLE>" } };
            nodeNumberer[HoleNode] = 0;

            jWriter.WritePropertyName("ContextGraph");
            ContextGraph.WriteJson(jWriter, nodeNumberer, dummyNodeLabeler,
                DeletedNode.SyntaxTree.GetLineSpan(DeletedNode.Span).StartLinePosition.Line);

            jWriter.WritePropertyName("HoleNode");
            jWriter.WriteValue(nodeNumberer[HoleNode]);

            jWriter.WritePropertyName("LastTokenBeforeHole");
            jWriter.WriteValue(nodeNumberer[LastTokenBeforeHole]);

            jWriter.WritePropertyName("LastUseOfVariablesInScope");
            jWriter.WriteStartObject();
            var droppedContextVariables = new HashSet<SyntaxToken>();
            foreach (var varNode in LastUseOfVariablesInScope)
            {
                if (!nodeNumberer.ContainsKey(varNode) || GetAllUses(varNode, IsOutsideOfDeletedNode).Where(t => t.Parent != null).Count() == 0)
                {
                    Console.WriteLine($"Extraction issue: Dropping variable {varNode} because we don't have its node.");
                    droppedContextVariables.Add(varNode);
                    continue; // May happen for symbols that are inherited or in partial classes.
                }
                jWriter.WritePropertyName(varNode.ToString());
                jWriter.WriteValue(GetNodeNumber(nodeNumberer, varNode));
            }
            jWriter.WriteEndObject();

            var productionNodes = new HashSet<ProducedSymbol>();
            jWriter.WritePropertyName("Productions");
            jWriter.WriteStartObject();
            foreach (var (nonterminal, production) in Productions)
            {
                jWriter.WritePropertyName(nonterminal.Id.ToString());
                productionNodes.Add(nonterminal);
                jWriter.WriteStartArray();
                foreach (var prodRhsNode in production)
                {
                    jWriter.WriteValue(prodRhsNode.Id);
                    productionNodes.Add(prodRhsNode);
                }
                jWriter.WriteEndArray();
            }
            jWriter.WriteEndObject();

            jWriter.WritePropertyName("SymbolKinds");
            jWriter.WriteStartObject();
            foreach (var node in productionNodes)
            {
                jWriter.WritePropertyName(node.Id.ToString());
                jWriter.WriteValue(node.SymbolKind.ToString());
            }
            jWriter.WriteEndObject();

            jWriter.WritePropertyName("SymbolLabels");
            jWriter.WriteStartObject();
            foreach (var node in productionNodes)
            {
                if (node.Label != null)
                {
                    jWriter.WritePropertyName(node.Id.ToString());
                    jWriter.WriteValue(node.Label.ToString());
                }
            }
            jWriter.WriteEndObject();

            WriteHoleTokenContext(jWriter);
            WriteVariableTokenContext(jWriter, nodeNumberer, droppedContextVariables);

            return (NodeNumberer: nodeNumberer, NodeLabeler: dummyNodeLabeler);
        }

        private void WriteHoleTokenContext(JsonWriter jWriter)
        {
            // Hole context tokens
            jWriter.WritePropertyName("HoleTokensBefore");
            WriteArrayWithSelfAndLeftTokenContext(jWriter, LastTokenBeforeHole);

            jWriter.WritePropertyName("HoleTokensAfter");
            WriteArrayWithSelfAndRightTokenContext(jWriter, DeletedNode.GetLastToken().GetNextToken());
        }

        private void WriteTokenArrayWithSelfAndContext(JsonWriter jWriter, SyntaxToken currentToken,
            Func<SyntaxToken, SyntaxToken> getNextToken, bool reverse = false)
        {
            var tokens = new List<SyntaxToken>();
            for (int i = 0; i < NUM_TOKENS_CONTEXT; i++)
            {
                tokens.Add(currentToken);
                currentToken = getNextToken(currentToken);
                if (currentToken == null) break;
            }

            jWriter.WriteStartArray();
            if (reverse) tokens.Reverse();
            foreach (var token in tokens)
            {
                jWriter.WriteStartArray();
                jWriter.WriteValue(token.Text);
                jWriter.WriteValue(ContextGraph.GetNodeType(token));
                jWriter.WriteEndArray();
            }
            jWriter.WriteEndArray();
        }

        private void WriteArrayWithSelfAndRightTokenContext(JsonWriter jWriter, SyntaxToken currentToken)
            => WriteTokenArrayWithSelfAndContext(jWriter, currentToken, t => t.GetNextToken());

        private void WriteArrayWithSelfAndLeftTokenContext(JsonWriter jWriter, SyntaxToken currentToken)
            => WriteTokenArrayWithSelfAndContext(jWriter, currentToken, t => t.GetPreviousToken(), reverse: true);

        private bool IsOutsideOfDeletedNode(SyntaxNodeOrToken tok) => !DeletedNode.DescendantNodesAndTokens().Contains(tok);

        private void WriteVariableTokenContext(JsonWriter jWriter, Dictionary<SyntaxNodeOrToken, int> NodeNumberer,
            HashSet<SyntaxToken> droppedContextVariables)
        {
            // First for each variable, find all its tokens not in the DeletedNode.            
            var allVariableUses = LastUseOfVariablesInScope.Except(droppedContextVariables)
                .ToDictionary(t => t, t => GetAllUses(t, IsOutsideOfDeletedNode));

            // Write results
            jWriter.WritePropertyName("VariableUsageContexts");
            jWriter.WriteStartArray();
            foreach (var (lastToken, allUses) in allVariableUses)
            {
                if (!NodeNumberer.ContainsKey(lastToken))
                {
                    continue;
                }
                jWriter.WriteStartObject();
                jWriter.WritePropertyName("NodeId");
                jWriter.WriteValue(GetNodeNumber(NodeNumberer, lastToken));
                jWriter.WritePropertyName("Name");
                jWriter.WriteValue(lastToken.Text);

                jWriter.WritePropertyName("TokenContexts");
                jWriter.WriteStartArray();
                int numUsesWritten = 0;
                foreach (var usedToken in allUses)
                {
                    if (usedToken.Parent == null)
                    {
                        // Happens for dummy node
                        continue;
                    }
                    jWriter.WriteStartArray();
                    WriteArrayWithSelfAndLeftTokenContext(jWriter, usedToken.GetPreviousToken());
                    WriteArrayWithSelfAndRightTokenContext(jWriter, usedToken.GetNextToken());
                    jWriter.WriteEndArray();
                    numUsesWritten++;
                }
                Debug.Assert(numUsesWritten > 0);
                jWriter.WriteEndArray();
                jWriter.WriteEndObject();
            }
            jWriter.WriteEndArray();
        }

        private IEnumerable<SyntaxToken> GetAllUses(SyntaxToken token, Predicate<SyntaxNodeOrToken> includeUse)
        {
            var allUses = new HashSet<SyntaxToken>();
            var toVisit = new Stack<SyntaxToken>();
            toVisit.Push(token);

            while (toVisit.Count > 0)
            {
                var currentToken = toVisit.Pop();
                allUses.Add(currentToken);
                if (includeUse(currentToken)) yield return currentToken;

                var (inEdges, outEdges) = ContextGraph.GetEdges(currentToken);
                foreach (var edge in inEdges.Concat(outEdges).Where(e => e.Label == SourceGraphEdge.LastUse || e.Label == SourceGraphEdge.LastLexicalUse))
                {
                    Debug.Assert(edge.Source.IsToken);
                    var sourceToken = edge.Source.AsToken();
                    if (!allUses.Contains(sourceToken))
                    {
                        toVisit.Push(sourceToken);
                    }

                    Debug.Assert(edge.Target.IsToken);
                    var targetToken = edge.Target.AsToken();
                    if (!allUses.Contains(targetToken))
                    {
                        toVisit.Push(targetToken);
                    }
                }
            }
        }
    }

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

        private readonly string _typeHierarchyOutputFilename;
        private readonly string _dotDebugDirectory;

        private readonly ChunkedJsonGzWriter _writer;
        private readonly Random _rng = new Random();

        private readonly string _repositoryRootPath;
        private readonly PersistentResumeList _fileResumeList;
        private readonly ThreadSafeTextLog _log;
        private readonly Action<SyntaxNode> _additionalExtractors;

        public TypeHierarchy TypeHierarchy { get; private set; }

        public SourceGraphExtractor(string repositoryRootPath, string graphOutputFilenameTemplate,
            string typeHierarchyOutputFilename, string dotDebugDir = null, string logPath = null,
            Action<SyntaxNode> additionalExtractors=null)
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

        private void WriteSample(HoleContextInformation sample)
        {
            _writer.WriteElement(jw => sample.WriteJson(jw));
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

        #region Expression hole-specific graph extraction
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


        private static void CopySubgraphAroundHole(
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

        public (SourceGraph contextGraph, SyntaxToken holeDummyNode, SyntaxToken tokenBeforeHole, List<SyntaxToken> variableInHoleDummyNodes) ExtractContextInformationForTargetNode(SemanticModel semanticModel, SyntaxToken[] allTokens, SourceGraph sourceGraph, SyntaxNode targetNode)
        {
            var holeNodes = new HashSet<SyntaxNodeOrToken>(targetNode.DescendantNodesAndTokensAndSelf());
            var firstHoleTokenIdx = Array.FindIndex(allTokens, t => holeNodes.Contains(t));
            var lastHoleTokenIdx = Array.FindLastIndex(allTokens, t => holeNodes.Contains(t));

            // Identify all symbols that are of types we consider and that are in scope:
            var symbolsInScope = RoslynUtils.GetAvailableValueSymbols(semanticModel, holeNodes.First(n => n.IsToken).AsToken());
            var typeLimitedSymbolsInScope = symbolsInScope.Where(sym => HasSimpleType(sym));

            /* // Debug code:
            var symbolTypes = symbolsInScope.Select(sym => TypeHierarchy.ComputeTypeForSymbol(sym, out ITypeSymbol symbolType) ? symbolType : null);
            Console.WriteLine($"Types:  {String.Join("  ", symbolTypes.Select(t => t?.ToString()))}");
            Console.WriteLine($"Expression: {targetNode}");
            var debug = typeLimitedSymbolsInScope.Select(sym => TypeHierarchy.ComputeTypeForSymbol(sym, out ITypeSymbol symbolType) ? (sym, symbolType) : (sym, null));
            Console.WriteLine($"Vars in Scope:  {String.Join(" ; ", debug.Select(symWithSymTyp => $"{symWithSymTyp.symbolType} {symWithSymTyp.sym.Name}"))}");
            */

            // Find all edges connecting the rest of the graph to the hole we are making:
            SyntaxToken tokenBeforeHole = default;
            SyntaxNodeOrToken lastUsedVariableBeforeHole = default;
            SyntaxNodeOrToken lastUsedVariableAfterHole = default;
            var variableToLastLexicalUse = new Dictionary<string, SyntaxNodeOrToken>();
            var variableToNextLexicalUse = new Dictionary<string, SyntaxNodeOrToken>();
            var dataflowSuccessorsOfHole = new HashSet<SyntaxNodeOrToken>();

            foreach (var holeNode in holeNodes)
            {
                var (inEdges, outEdges) = sourceGraph.GetEdges(holeNode);
                foreach (var (otherNode, label, _) in inEdges)
                {
                    // We can safely ignore these, because we don't need to update anything related to them:
                    if (label == SourceGraphEdge.Child) continue;

                    // Look only at edges crossing the boundary between hole/context:
                    if (!holeNodes.Contains(otherNode))
                    {
                        if (label == SourceGraphEdge.NextToken)
                        {
                            tokenBeforeHole = otherNode.AsToken();  // This is the lexical predecessor of the hole.
                        }
                        else if (label == SourceGraphEdge.LastUsedVariable)
                        {
                            lastUsedVariableAfterHole = otherNode;  // This is the variable use successor of the hole
                        }
                        else if (label == SourceGraphEdge.LastUse || label == SourceGraphEdge.LastWrite)
                        {
                            dataflowSuccessorsOfHole.Add(otherNode);  // This node's dataflow is influenced by the hole
                        }
                    }
                }
                foreach (var (_, label, otherNode) in outEdges)
                {
                    // We can safely ignore these, because we don't need to update anything related to them:
                    if (label == SourceGraphEdge.Child) continue;

                    // Look only at edges crossing the boundary between hole/context:
                    if (!holeNodes.Contains(otherNode))
                    {
                        if (label == SourceGraphEdge.LastUsedVariable)
                        {
                            lastUsedVariableBeforeHole = otherNode;  // This is the variable use predecessor of the hole
                        }
                    }
                }
            }

            // Now to the actual copying. First, add a dummy node that will stand in for the expression, and then just copy everything "near" the hole, but nothing in the hole.
            var resultGraph = SourceGraph.Create(sourceGraph);
            var holeDummyNode = SyntaxFactory.Identifier("%HOLE%");
            var targetNodeType = semanticModel.GetTypeInfo(targetNode).Type;
            resultGraph.OverrideTypeForNode(holeDummyNode, targetNodeType?.ToString() ?? "?");
            resultGraph.AddEdge(targetNode.Parent, SourceGraphEdge.Child, holeDummyNode);
            resultGraph.AddEdge(allTokens[firstHoleTokenIdx - 1], SourceGraphEdge.NextToken, holeDummyNode);
            resultGraph.AddEdge(holeDummyNode, SourceGraphEdge.NextToken, allTokens[lastHoleTokenIdx + 1]);
            //Make sure that we have all the important nodes in the graph:
            var (lastTokeninEdges, lastTokenoutEdges) = sourceGraph.GetEdges(tokenBeforeHole);
            foreach (var edge in lastTokeninEdges.Concat(lastTokenoutEdges)) { resultGraph.AddEdge(edge); }
            CopySubgraphAroundHole(sourceGraph, holeNodes, holeNodes, resultGraph);

            // Now add dummy nodes for all variables that we may need to consider:
            var symbolDummyTokenList = new List<SyntaxToken>();
            var edgesToRemoveInResult = new List<Edge<SyntaxNodeOrToken, SourceGraphEdge>>();
            int numSymbolsAddedSoFar = 0;
            foreach (var symbolInScope in typeLimitedSymbolsInScope)
            {
                numSymbolsAddedSoFar++;
                var symbolDummyNode = SyntaxFactory.Identifier(symbolInScope.Name);
                if (RoslynUtils.GetTypeSymbol(symbolInScope, out var symbolInScopeType))
                {
                    resultGraph.OverrideTypeForNode(symbolDummyNode, symbolInScopeType.ToString());
                }
                else
                {
                    resultGraph.OverrideTypeForNode(symbolDummyNode, SourceGraph.NOTYPE_NAME);
                }

                // Step 1: Insert edges linking the dummy node to the context surrounding the hole:
                if (lastUsedVariableBeforeHole != default)
                {
                    sourceGraph.AddEdge(symbolDummyNode, SourceGraphEdge.LastUsedVariable, lastUsedVariableBeforeHole);
                }
                if (lastUsedVariableAfterHole != default)
                {
                    sourceGraph.AddEdge(lastUsedVariableAfterHole, SourceGraphEdge.LastUsedVariable, symbolDummyNode);
                }

                var newEdges = new List<Edge<SyntaxNodeOrToken, SourceGraphEdge>>();
                var removedEdges = new List<Edge<SyntaxNodeOrToken, SourceGraphEdge>>();
                try
                {
                    //Step 2: Insert new dataflow edges from dummy node
                    DataFlowGraphHelper.AddDataFlowEdges(sourceGraph, symbolDummyNode, forbiddenNodes: holeNodes, addedEdges: newEdges);

                    //Step 3: Update dataflow edges from things that follow hole
                    //We use that nodes whose dataflow depends on the new nodes would have been connected to the nodes used as targets of our new dummy node's dataflow edges.
                    //Thus, we look at their incoming edges, and recompute dataflow for the sources of those:
                    var possiblyInfluencedNodes = new HashSet<SyntaxNodeOrToken>();
                    foreach (var newEdge in newEdges)
                    {
                        foreach (var possiblyChangedEdge in sourceGraph.GetInEdges(newEdge.Target))
                        {
                            if (possiblyChangedEdge.Source != symbolDummyNode
                                && (possiblyChangedEdge.Label == SourceGraphEdge.LastUse || possiblyChangedEdge.Label == SourceGraphEdge.LastWrite))
                            {
                                removedEdges.Add(possiblyChangedEdge);
                                possiblyInfluencedNodes.Add(possiblyChangedEdge.Source);
                            }
                        }
                    }

                    //Step 4: Update lexical use information. Find previous and last uses:
                    int prevIdx = Array.FindLastIndex(allTokens,
                        firstHoleTokenIdx - 1,
                        curTok => curTok.Text.Equals(symbolInScope.Name) && sourceGraph.UsedVariableNodes.Contains(curTok));
                    if (prevIdx > 0)
                    {
                        var prevTok = allTokens[prevIdx];
                        foreach (var inEdge in sourceGraph.GetInEdges(prevTok))
                        {
                            if (inEdge.Label == SourceGraphEdge.LastLexicalUse)
                            {
                                removedEdges.Add(inEdge);
                                break;
                            }
                        }
                        sourceGraph.AddEdge(symbolDummyNode, SourceGraphEdge.LastLexicalUse, prevTok);
                    }
                    int nextIdx = Array.FindIndex(allTokens,
                        lastHoleTokenIdx + 1,
                        curTok => curTok.Text.Equals(symbolInScope.Name) && sourceGraph.UsedVariableNodes.Contains(curTok));
                    if (nextIdx > 0)
                    {
                        var nextTok = allTokens[nextIdx];
                        foreach (var outEdge in sourceGraph.GetOutEdges(nextTok))
                        {
                            if (outEdge.Label == SourceGraphEdge.LastLexicalUse)
                            {
                                removedEdges.Add(outEdge);
                                break;
                            }
                        }
                        sourceGraph.AddEdge(nextTok, SourceGraphEdge.LastLexicalUse, symbolDummyNode);
                    }

                    //Step 5: Temporarily commit changes to graph:
                    foreach (var edgeToRemove in removedEdges)
                    {
                        sourceGraph.RemoveEdge(edgeToRemove);
                    }

                    foreach (var changedNode in possiblyInfluencedNodes.Concat(dataflowSuccessorsOfHole))
                    {
                        DataFlowGraphHelper.AddDataFlowEdges(sourceGraph, changedNode, forbiddenNodes: holeNodes, addedEdges: newEdges);
                    }

                    //Extract the actual subgraph around that dummy node:
                    CopySubgraphAroundHole(
                        sourceGraph,
                        new SyntaxNodeOrToken[] { symbolDummyNode },
                        holeNodes,
                        resultGraph,
                        numSymbolsAddedSoFar);
                    symbolDummyTokenList.Add(symbolDummyNode);
                }
                finally
                {
                    //Roll back changes to the source graph:
                    foreach (var newEdge in newEdges)
                    {
                        sourceGraph.RemoveEdge(newEdge);
                    }
                    foreach (var removedEdge in removedEdges)
                    {
                        edgesToRemoveInResult.Add(removedEdge);
                        sourceGraph.AddEdge(removedEdge);
                    }
                    //Remove the dummy node we introduced:
                    sourceGraph.RemoveNode(symbolDummyNode);
                }
            }

            foreach (var edge in edgesToRemoveInResult)
            {
                resultGraph.RemoveEdge(edge);
            }

            return (resultGraph, holeDummyNode, tokenBeforeHole, symbolDummyTokenList);
        }
        #endregion

        public void ExtractAllSamplesFromSemanticModel(object sender, SemanticModel semanticModel)
        {
            var tree = semanticModel.SyntaxTree;
            if (_fileResumeList.AddIfNotContained(tree.FilePath))
            {
                int numExpressionsExtracted = 0;
                try
                {
                    if (tree.FilePath.Contains("TemporaryGeneratedFile") || tree.FilePath.EndsWith(".Generated.cs"))
                    {
                        Console.WriteLine($"Ignoring file {tree.FilePath}.");
                        return; // TODO: Remove any temp files in obj
                    }
                    //Console.WriteLine($"Writing out samples for {tree.FilePath}");
                    
                    var allTokens = tree.GetRoot().DescendantTokens().Where(t => t.Text.Length > 0).ToArray();

                    var addedTypes = new HashSet<ITypeSymbol>();
                    foreach (var syntaxNode in tree.GetRoot().DescendantNodes())
                    {
                        var symbol = RoslynUtils.GetReferenceSymbol(syntaxNode, semanticModel);
                        if (RoslynUtils.GetTypeSymbol(symbol, out var typeSymbol) && addedTypes.Add(typeSymbol))
                        {
                            TypeHierarchy.Add(typeSymbol);
                        }
                    }

                    SourceGraph sourceGraph = null;
                    foreach (var simpleExpressionNode in SimpleExpressionIdentifier.GetSimpleExpressions(semanticModel))
                    {
                        try
                        {
                            if (sourceGraph == null)
                            {
                                sourceGraph = ExtractSourceGraph(semanticModel, allTokens); // Compute the graph if we haven't done so yet...
                                                                                            //sourceGraph.ToDotFile(Path.GetFileName(tree.FilePath) + ".graph.dot", sourceGraph.GetNodeNumberer(), new Dictionary<SyntaxNodeOrToken, string>(), false);
                            }
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine($"Error while extracting graph. Aborting file. Cause: {e.Message}");
                            return;
                        }

                        var (contextGraph, holeDummyNode, tokenBeforeHole, variableNodesInScope) = ExtractContextInformationForTargetNode(semanticModel, allTokens, sourceGraph, simpleExpressionNode);

                        var allVariablesInExpression = RoslynUtils.GetUsedVariableSymbols(semanticModel, simpleExpressionNode, onlyLocalFileVariables: false);
                        if (!allVariablesInExpression.Any())
                        {
                            // We filter nodes that do not contain any variables.
                            continue;
                        }

                        bool isNotInCurrentFile(ISymbol symbol)
                        {
                            var symbolLocation = symbol.Locations.First();
                            return symbolLocation.IsInMetadata || symbolLocation.SourceTree.FilePath != tree.FilePath;
                        };
                        if (allVariablesInExpression.Count == 0 || allVariablesInExpression.Any(isNotInCurrentFile))
                        {
                            continue;
                        }

                        var productions = TurnIntoProductions(simpleExpressionNode).ToList();
                        // Debug:
                        /*
                        Console.WriteLine($"Simple Expression: {simpleExpressionNode}");
                        foreach (var (nonterm, productionRhs) in productions)
                        {
                            var productionStrings = 
                                productionRhs.Select(t => (t.SymbolKind == ExpansionSymbolKind.Token) ? t.Label.ToString() : t.SymbolKind.ToString()
                                                          + "[" + t.Id + "]"
                                                          + ((t.Label != null) ? "(" + t.Label + ")" : ""));
                            Console.WriteLine($"  {nonterm.SymbolKind.ToString()}[{nonterm.Id}] --> {String.Join("  ", productionStrings)}");
                        }
                        */

                        var sample = new HoleContextInformation(_repositoryRootPath, simpleExpressionNode, contextGraph, holeDummyNode, tokenBeforeHole, variableNodesInScope, productions);
                        WriteSample(sample);
                        numExpressionsExtracted++;

                        // Optionally invoke additional extractor.
                        _additionalExtractors?.Invoke(simpleExpressionNode);
                    }
                    if (numExpressionsExtracted > 0)
                    {
                        _log.LogMessage($"Extracted {numExpressionsExtracted} expressions from {tree.FilePath}.");
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Exception when extracting data from file: {e.Message}: {e.StackTrace}");
                    _log.LogMessage($"Error{e.Message}: {e.StackTrace} while extracting. Managed to extract {numExpressionsExtracted} expressions from {tree.FilePath}.");
                }
                if (_rng.NextDouble() > .9)
                {
                    // Once in a while save they type hierarchy
                    TypeHierarchy.SaveTypeHierarchy(_typeHierarchyOutputFilename);
                }
            }
        }

        

        private IEnumerable<(ProducedSymbol nonTerminal, IEnumerable<ProducedSymbol> productionRhs)> TurnIntoProductions(SyntaxNode node)
        {
            var toVisit =
                new Stack<(SyntaxNodeOrToken astNode, bool requiresExtraStepForTerminal, ProducedSymbol generatedSymbol)>();

            uint nextSymbolId = 0;

            toVisit.Push((node, true, new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Expression, null)));

            /*
             * TODO: Redo this.
             *   Target grammar:
             *     Expr ->   Expr + Expr | Expr <= Expr
             *             | Expr [ Expr ]
             *             | Expr . Length
             *             | InvocationExpr
             *     InvocationExpr ->   Expression . StartsWith ( Expr )
             *                       | Expression . IndexOf ( Expr )
             *                       | Expression . IndexOf ( Expr, Expr )
             *                       | ...
             */

            while (toVisit.Count > 0)
            {
                var (nonterminalNode, requiresExtraStepForTerminal, generatedSymbol) = toVisit.Pop();

                var additionalProductions = new List<(ProducedSymbol nonTerminal, IEnumerable<ProducedSymbol> production)>();
                var productionRhs = new List<ProducedSymbol>();
                var children = nonterminalNode.ChildNodesAndTokens().ToArray();

                // Directly collapse "Expression ArgumentListExpression" into one production:
                if (nonterminalNode.AsNode() is InvocationExpressionSyntax invocationNode)
                {
                    SyntaxNodeOrToken[] expressionNodes;
                    if (invocationNode.Expression is MemberAccessExpressionSyntax memberAccess)
                    {
                        SyntaxNodeOrToken accessedMember;
                        if (memberAccess.Name is IdentifierNameSyntax idNode)
                        {
                            accessedMember = idNode.Identifier;
                        }
                        else
                        {
                            accessedMember = memberAccess.Name;
                        }
                        expressionNodes = new SyntaxNodeOrToken[] { memberAccess.Expression,
                            memberAccess.OperatorToken,
                            accessedMember };
                    }
                    else
                    {
                        expressionNodes = new SyntaxNodeOrToken[] { invocationNode.Expression };
                    }
                    var argumentNodes =
                        invocationNode.ArgumentList.ChildNodesAndTokens()
                                                   .Select(childNode => childNode.AsNode() is ArgumentSyntax argumentChild ? argumentChild.Expression : childNode);
                    children = expressionNodes.Concat(argumentNodes).ToArray();
                }

                for (int i = 0; i < children.Length; ++i)
                {
                    var childNode = children[i];
                    if (childNode.IsKind(SyntaxKind.NumericLiteralToken)
                             || childNode.IsKind(SyntaxKind.StringLiteralToken)
                             || childNode.IsKind(SyntaxKind.CharacterLiteralToken)
                             || childNode.IsKind(SyntaxKind.NumericLiteralExpression)
                             || childNode.IsKind(SyntaxKind.StringLiteralExpression)
                             || childNode.IsKind(SyntaxKind.CharacterLiteralExpression))
                    {
                        object literalValue;
                        if (childNode.IsToken)
                        {
                            literalValue = childNode.AsToken().Value;
                        }
                        else
                        {
                            literalValue = (childNode.AsNode() as LiteralExpressionSyntax).ToString();
                        }
                        ExpansionSymbolKind literalKind;
                        if (childNode.IsKind(SyntaxKind.NumericLiteralExpression) || childNode.IsKind(SyntaxKind.NumericLiteralToken))
                        {
                            literalKind = ExpansionSymbolKind.IntLiteral;
                        }
                        else if (childNode.IsKind(SyntaxKind.StringLiteralExpression) || childNode.IsKind(SyntaxKind.StringLiteralToken))
                        {
                            literalKind = ExpansionSymbolKind.StringLiteral;
                        }
                        else if (childNode.IsKind(SyntaxKind.CharacterLiteralExpression) || childNode.IsKind(SyntaxKind.CharacterLiteralToken))
                        {
                            literalKind = ExpansionSymbolKind.CharLiteral;
                        }
                        else
                        {
                            throw new NotImplementedException("Unknown literal type encountered!");
                        }
                        var newNonterminalSymbol = requiresExtraStepForTerminal ? new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Expression, null) : null;
                        var newLiteralSymbol = new ProducedSymbol(++nextSymbolId, literalKind, literalValue);
                        if (requiresExtraStepForTerminal)
                        {
                            productionRhs.Add(newNonterminalSymbol);
                            additionalProductions.Add((newNonterminalSymbol, new[] { newLiteralSymbol }));
                        }
                        else
                        {
                            productionRhs.Add(newLiteralSymbol);
                        }
                    }
                    else if (childNode.IsToken)
                    {
                        var newTokenSymbol = new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Token, childNode.AsToken().ValueText);
                        productionRhs.Add(newTokenSymbol);
                    }
                    else if (childNode.IsKind(SyntaxKind.IdentifierName))
                    {
                        // Special case for MemberAccesses (calls & properties), where we only have a limited set of allowed names and just add them as tokens:
                        if (nonterminalNode.IsKind(SyntaxKind.SimpleMemberAccessExpression) && i == children.Length - 1)
                        {
                            var newTokenSymbol = new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Token, childNode.ToString());
                            productionRhs.Add(newTokenSymbol);
                        }
                        else
                        {
                            var newNonterminalSymbol = requiresExtraStepForTerminal ? new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Expression, null) : null;
                            var identifierName = (childNode.AsNode() as IdentifierNameSyntax).Identifier.Text;
                            // Stip "@" off of masked keywords, as we use proper symbols in the rest of the code, which don't have the @ either:
                            if (identifierName.StartsWith("@"))
                            {
                                identifierName = identifierName.Substring(1);
                            }
                            var newVariableSymbol = new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Variable, identifierName);
                            if (requiresExtraStepForTerminal)
                            {
                                productionRhs.Add(newNonterminalSymbol);
                                additionalProductions.Add((newNonterminalSymbol, new[] { newVariableSymbol }));
                            }
                            else
                            {
                                productionRhs.Add(newVariableSymbol);
                            }
                        }
                    }
                    else if (childNode.IsKind(SyntaxKind.BracketedArgumentList))
                    {
                        // Special case to collapse nested AST steps for singleton argument lists:
                        var argumentListNode = childNode.AsNode() as BracketedArgumentListSyntax;
                        productionRhs.Add(new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Token, "["));
                        var argumentExprSymbol = new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Expression, null);
                        productionRhs.Add(argumentExprSymbol);
                        productionRhs.Add(new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Token, "]"));
                        toVisit.Push((argumentListNode.Arguments[0], false, argumentExprSymbol));
                    }
                    else
                    {
                        var newNonterminalSymbol = new ProducedSymbol(++nextSymbolId, ExpansionSymbolKind.Expression, null);
                        productionRhs.Add(newNonterminalSymbol);
                        toVisit.Push((childNode, true, newNonterminalSymbol));
                    }
                }

                yield return (generatedSymbol, productionRhs);

                foreach (var production in additionalProductions)
                {
                    yield return production;
                }
            }
        }
    }

    /*
     * Simple whitelisting visitor pattern, marking up nodes in the tree which only have children that
     * we approve of.
     */
    public class SimpleExpressionIdentifier : CSharpSyntaxWalker
    {
        private readonly Dictionary<SyntaxNode, bool> IsSimpleNode;
        private readonly SemanticModel SemanticModel;

        public static IEnumerable<ExpressionSyntax> GetSimpleExpressions(SemanticModel semanticModel)
        {
            var visitor = new SimpleExpressionIdentifier(semanticModel);
            visitor.Visit(semanticModel.SyntaxTree.GetRoot());

            foreach (var (node, nodeIsSimple) in visitor.IsSimpleNode)
            {
                //Only return the topmost simple expression:
                if (nodeIsSimple && !visitor.IsSimpleNode[node.Parent] && !(node is ArgumentListSyntax))
                {
                    //Special case 1: If we are an argument, the argument list is not simple (checked above), skip this one (instead, exports the encapsulated expression):
                    if (node is ArgumentSyntax && node.Parent is ArgumentListSyntax && !visitor.IsSimpleNode[node.Parent.Parent])
                    {
                        continue;
                    }
                    //Special case 2: If we are a member access in an Invocation, and our parent is /not/ simple, then one of the arguments isn't simple, so skip the member access:
                    if (node is MemberAccessExpressionSyntax && node.Parent is InvocationExpressionSyntax)
                    {
                        continue;
                    }

                    //Also filter out AST leafs:
                    if (node.ChildNodes().Any() && node is ExpressionSyntax expressionNode)
                    {
                        yield return expressionNode;
                    }
                }
            }
        }

        private SimpleExpressionIdentifier(SemanticModel semanticModel)
        {
            IsSimpleNode = new Dictionary<SyntaxNode, bool>();
            SemanticModel = semanticModel;
        }

        public override void DefaultVisit(SyntaxNode node)
        {
            // First, visit the children:
            base.DefaultVisit(node);
            // By default, all nodes are non-simple.
            IsSimpleNode[node] = false;
        }

        private void SimpleVisit(SyntaxNode node)
        {
            // First, visit the children:
            base.DefaultVisit(node);
            // We are only in here if we approve of the current node type, now check if all children are fine, too:
            IsSimpleNode[node] = node.ChildNodes().All(childNode => IsSimpleNode[childNode]);
        }

        public override void VisitParenthesizedExpression(ParenthesizedExpressionSyntax node) => SimpleVisit(node);
        public override void VisitTupleExpression(TupleExpressionSyntax node) => SimpleVisit(node);
        public override void VisitPrefixUnaryExpression(PrefixUnaryExpressionSyntax node) => SimpleVisit(node);
        public override void VisitPostfixUnaryExpression(PostfixUnaryExpressionSyntax node) => SimpleVisit(node);
        public override void VisitBinaryExpression(BinaryExpressionSyntax node) => SimpleVisit(node);
        public override void VisitConditionalExpression(ConditionalExpressionSyntax node) => SimpleVisit(node);
        //public override void VisitDefaultExpression(DefaultExpressionSyntax node) => SimpleVisit(node);
        //public override void VisitTypeOfExpression(TypeOfExpressionSyntax node) => SimpleVisit(node);
        //public override void VisitSizeOfExpression(SizeOfExpressionSyntax node) => SimpleVisit(node);
        public override void VisitElementAccessExpression(ElementAccessExpressionSyntax node) => SimpleVisit(node);
        public override void VisitArgument(ArgumentSyntax node) => SimpleVisit(node);
        public override void VisitInvocationExpression(InvocationExpressionSyntax node) => SimpleVisit(node);
        public override void VisitArgumentList(ArgumentListSyntax node) => SimpleVisit(node);

        public override void VisitBracketedArgumentList(BracketedArgumentListSyntax node)
        {
            // First, think of the children:
            base.DefaultVisit(node);
            // We are simple if our argument is simple:
            IsSimpleNode[node] = (node.Arguments.Count == 1) && IsSimpleNode[node.Arguments[0]];
        }

        public override void VisitMemberAccessExpression(MemberAccessExpressionSyntax node)
        {
            // First, visit all children:
            base.DefaultVisit(node);

            var expressionIsSimple = IsSimpleNode[node.Expression];

            if (!expressionIsSimple)
            {
                IsSimpleNode[node] = false;
                return;
            }

            // Check what the accessed type is, and allow accesses on strings:
            var accessedSymbol = SemanticModel.GetSymbolInfo(node.Expression).Symbol;
            if (TypeHierarchy.ComputeTypeForSymbol(accessedSymbol, out ITypeSymbol accessedTypeSymbol)
                && "String".Equals(accessedTypeSymbol.Name))
            {
                // If this is a call on a System.String method that returns an allowed type, allow it:
                if (node.Parent is InvocationExpressionSyntax)
                {
                    IsSimpleNode[node] = SourceGraphExtractor.IsSimpleType(SemanticModel.GetTypeInfo(node.Parent).Type);
                }
                // Access to string length is fine as well:
                else if (node.Name is IdentifierNameSyntax idNode
                    && "Length".Equals(idNode.Identifier.ValueText))
                {
                    IsSimpleNode[node] = true;
                }
                else
                {
                    IsSimpleNode[node] = false;
                }
            }
            // Mark (simpleArray).Length as simple as well:
            else if (node.Name is IdentifierNameSyntax idNode
                && "Length".Equals(idNode.Identifier.ValueText))
            {
                IsSimpleNode[node] = true;
            }
            else
            {
                IsSimpleNode[node] = false;
            }
        }

        public override void VisitIdentifierName(IdentifierNameSyntax node)
        {
            // Check if the type of the identifier is acceptably simple:
            var nodeSymbol = SemanticModel.GetSymbolInfo(node);
            IsSimpleNode[node] = SourceGraphExtractor.HasSimpleType(nodeSymbol.Symbol);
        }

        public override void VisitLiteralExpression(LiteralExpressionSyntax node)
        {
            IsSimpleNode[node] = RoslynUtils.IsSimpleLiteral(node);
        }
    }
}
