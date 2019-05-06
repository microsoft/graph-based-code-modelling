using DataExtraction.Utils;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System.Diagnostics;
using System.Linq;

namespace SourceGraphExtractionUtils.Utils
{
    public enum SourceGraphEdge
    {
        Child,
        NextToken,
        LastUsedVariable,
        LastUse,
        LastWrite,
        LastLexicalUse,
        ComputedFrom,
        ReturnsTo,
        FormalArgName,
        GuardedBy,
        GuardedByNegation,
        BindsToSymbol
    }

    public class SourceGraphComparer
        : IEqualityComparer<Edge<SyntaxNodeOrToken, SourceGraphEdge>>,
          IEqualityComparer<SyntaxNodeOrToken>
    {
        bool IEqualityComparer<Edge<SyntaxNodeOrToken, SourceGraphEdge>>.Equals(Edge<SyntaxNodeOrToken, SourceGraphEdge> x, Edge<SyntaxNodeOrToken, SourceGraphEdge> y)
            => x.Label == y.Label
                && ((IEqualityComparer<SyntaxNodeOrToken>)this).Equals(x.Source, y.Source)
                && ((IEqualityComparer<SyntaxNodeOrToken>)this).Equals(x.Target, y.Target);

        bool IEqualityComparer<SyntaxNodeOrToken>.Equals(SyntaxNodeOrToken x, SyntaxNodeOrToken y)
            => x.Equals(y);

        int IEqualityComparer<Edge<SyntaxNodeOrToken, SourceGraphEdge>>.GetHashCode(Edge<SyntaxNodeOrToken, SourceGraphEdge> edge)
            => 29 * edge.Source.GetHashCode() + 31 * edge.Target.GetHashCode() + 37 * edge.Label.GetHashCode();

        int IEqualityComparer<SyntaxNodeOrToken>.GetHashCode(SyntaxNodeOrToken node)
            => node.GetHashCode();
    }

    public class SourceGraph : DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge>
    {
        public const string NOTYPE_NAME = "NOTYPE";

        public readonly SemanticModel SemanticModel;
        private readonly Dictionary<SyntaxNodeOrToken, string> _nodeToTypeStringCache;
        private readonly HashSet<SyntaxNodeOrToken> _declarationNodes;
        private readonly HashSet<SyntaxNodeOrToken> _usedVariableNodes;
        private readonly HashSet<SyntaxNodeOrToken> _writtenVariableNodes;
        private readonly Dictionary<SyntaxToken, ISet<ISymbol>> _aliasedVariableNodes;

        private SourceGraph(SemanticModel semanticModel, SourceGraphComparer comparer) : base(comparer, comparer)
        {
            SemanticModel = semanticModel;
            _nodeToTypeStringCache = new Dictionary<SyntaxNodeOrToken, string>();
            _usedVariableNodes = new HashSet<SyntaxNodeOrToken>();
            _writtenVariableNodes = new HashSet<SyntaxNodeOrToken>();
            _declarationNodes = new HashSet<SyntaxNodeOrToken>();
            _aliasedVariableNodes = new Dictionary<SyntaxToken, ISet<ISymbol>>();
        }

        private SourceGraph(SourceGraph baseGraph, SourceGraphComparer comparer) : base(comparer, comparer)
        {
            SemanticModel = baseGraph.SemanticModel;
            _nodeToTypeStringCache = baseGraph._nodeToTypeStringCache;
            _usedVariableNodes = new HashSet<SyntaxNodeOrToken>(baseGraph._usedVariableNodes);
            _writtenVariableNodes = new HashSet<SyntaxNodeOrToken>(baseGraph._writtenVariableNodes);
            _declarationNodes = new HashSet<SyntaxNodeOrToken>(baseGraph._declarationNodes);
            _aliasedVariableNodes = new Dictionary<SyntaxToken, ISet<ISymbol>>();
            foreach (var kvp in baseGraph._aliasedVariableNodes)
            {
                _aliasedVariableNodes.Add(kvp.Key, new HashSet<ISymbol>(kvp.Value));
            }
        }

        public static SourceGraph Create(SemanticModel semanticModel)
        {
            var comparer = new SourceGraphComparer();
            return new SourceGraph(semanticModel, comparer);
        }

        public static SourceGraph Create(SourceGraph baseGraph)
        {
            var comparer = new SourceGraphComparer();
            return new SourceGraph(baseGraph, comparer);
        }

        public bool IsDeclaration(SyntaxNodeOrToken node) => _declarationNodes.Contains(node);

        public bool IsWrite(SyntaxNodeOrToken node) => _writtenVariableNodes.Contains(node);

        public ISet<SyntaxNodeOrToken> DeclarationNodes => _declarationNodes;

        public ISet<SyntaxNodeOrToken> UsedVariableNodes => _usedVariableNodes;

        public ISet<SyntaxNodeOrToken> WrittenVariableNodes => _writtenVariableNodes;

        public IDictionary<SyntaxToken, ISet<ISymbol>> AliasedVariableNodes => _aliasedVariableNodes;

        public string GetNodeLabel(SyntaxNodeOrToken node, Dictionary<SyntaxNodeOrToken, string> nodeLabelOverrides = null)
        {
            if (nodeLabelOverrides != null && nodeLabelOverrides.TryGetValue(node, out var label))
            {
                return label;
            }

            if (node.IsToken)
            {
                label = node.ToString();
                // Stip "@" off of masked keywords, as we use proper symbols in the rest of the code, which don't have the @ either:
                if (label.StartsWith("@"))
                {
                    label = label.Substring(1);
                }
                return label;
            }
            else
            {
                return node.AsNode().Kind().ToString();
            }
        }

        private Action<JsonWriter, Dictionary<SyntaxNodeOrToken, int>> WriteNodeLabelsJson(Dictionary<SyntaxNodeOrToken, string> nodeLabelOverrides)
            => (jWriter, nodeNumberer) =>
            {
                jWriter.WritePropertyName("NodeLabels");

                jWriter.WriteStartObject();
                foreach ((var node, var nodeNumber) in nodeNumberer)
                {
                    jWriter.WritePropertyName(nodeNumber.ToString());
                    jWriter.WriteValue(GetNodeLabel(node, nodeLabelOverrides));
                }
                jWriter.WriteEndObject();
            };

        public string GetNodeType(SyntaxNodeOrToken node)
        {
            if (!_nodeToTypeStringCache.TryGetValue(node, out var res))
            {
                // Handle some literal types:
                if (node.IsKind(SyntaxKind.StringLiteralToken))
                {
                    res = "string";
                }
                else if (node.IsKind(SyntaxKind.CharacterLiteralToken))
                {
                    res = "char";
                }
                else if (node.IsKind(SyntaxKind.NumericLiteralToken))
                {
                    res = node.AsToken().Value.GetType().Name.ToLower();
                }
                else
                {
                    var syntaxNode = node.IsNode ? node.AsNode() : node.AsToken().Parent;
                    if (syntaxNode != null)
                    {
                        ISymbol symbol = RoslynUtils.GetReferenceSymbol(syntaxNode, SemanticModel);
                        if (RoslynUtils.GetTypeSymbol(symbol, out var typeSymbol))
                        {
                            res = typeSymbol.ToString();
                        }
                        else
                        {
                            res = NOTYPE_NAME;
                        }
                    }
                    else
                    {
                        res = NOTYPE_NAME;
                    }
                }
                _nodeToTypeStringCache[node] = res;
            }
            return res;
        }

        public void OverrideTypeForNode(SyntaxNodeOrToken node, string type)
        {
            _nodeToTypeStringCache[node] = type;
        }

        private Action<JsonWriter, Dictionary<SyntaxNodeOrToken, int>> WriteNodeTypesJson()
            => (jWriter, nodeNumberer) =>
            {
                jWriter.WritePropertyName("NodeTypes");

                jWriter.WriteStartObject();
                foreach ((var node, var nodeNumber) in nodeNumberer)
                {
                    var nodeType = GetNodeType(node);
                    if (nodeType != NOTYPE_NAME)
                    {
                        jWriter.WritePropertyName(nodeNumber.ToString());
                        jWriter.WriteValue(nodeType);
                    }
                }
                jWriter.WriteEndObject();
            };

        private Action<JsonWriter, (SyntaxNodeOrToken From, SyntaxNodeOrToken To)> WriteEdgeDistanceValueJson(SemanticModel semanticModel, int slotLineNumber) =>
            (jWriter, edgeInfo) =>
            {
                int fromLineNo = slotLineNumber;
                if (edgeInfo.From.Parent != null)
                {
                    fromLineNo = edgeInfo.From.Parent.SyntaxTree.GetLineSpan(edgeInfo.From.FullSpan).StartLinePosition.Line;
                }
                int toLineNo = slotLineNumber;
                if (edgeInfo.To.Parent != null)
                {
                    toLineNo = edgeInfo.To.Parent.SyntaxTree.GetLineSpan(edgeInfo.To.FullSpan).EndLinePosition.Line;
                }
                jWriter.WriteValue(toLineNo - fromLineNo);
            };

        public void WriteJson(
            JsonWriter jWriter,
            Dictionary<SyntaxNodeOrToken, int> nodeNumberer,
            Dictionary<SyntaxNodeOrToken, string> nodeLabelOverrides,
            int? holeLineNumber = null)
        {
            var edgeDistanceInfoWriters = 
                holeLineNumber.HasValue
                ? new (Predicate<SourceGraphEdge> acceptsEdgeLabel,
                       Action<JsonWriter, (SyntaxNodeOrToken From, SyntaxNodeOrToken To)> writer)[]
                  {
                      (edgeType => edgeType.Equals(SourceGraphEdge.LastLexicalUse)
                                   || edgeType.Equals(SourceGraphEdge.LastUse)
                                   || edgeType.Equals(SourceGraphEdge.LastWrite),
                       WriteEdgeDistanceValueJson(SemanticModel, holeLineNumber.Value))
                  }
                : new (Predicate<SourceGraphEdge> acceptsEdgeLabel, 
                       Action<JsonWriter, (SyntaxNodeOrToken From, SyntaxNodeOrToken To)> writer)[0];
            WriteJson(jWriter,
                nodeNumberer,
                new Action<JsonWriter, Dictionary<SyntaxNodeOrToken, int>>[] {
                    WriteNodeLabelsJson(nodeLabelOverrides),
                    WriteNodeTypesJson(),
                },
                edgeDistanceInfoWriters);
        }

        public void ToDotFile(string outputPath, Dictionary<SyntaxNodeOrToken, int> nodeNumberer, Dictionary<SyntaxNodeOrToken, string> nodeLabeler, bool diffable=false)
        {
            Func<SyntaxNodeOrToken, object> ToLineSpan = t =>
            {
                var span = t.GetLocation().GetMappedLineSpan();
                return $"{span.StartLinePosition} -- {span.EndLinePosition}";
            };

            ToDotFile(outputPath, diffable ? nodeNumberer.ToDictionary(kv => kv.Key, kv => ToLineSpan(kv.Key)) 
                                           : nodeNumberer.ToDictionary(kv => kv.Key, kv => (object)kv.Value),
                      node => node.IsToken ? "rectangle" : "circle",
                      node => GetNodeLabel(node, nodeLabeler).Replace("\"", "\\\""));
        }
        
    }

    public class ASTGraphExtractor : CSharpSyntaxWalker
    {
        private SourceGraph _graph;
        private SyntaxToken? _lastAddedToken;

        private ASTGraphExtractor(SemanticModel semanticModel)
        {
            _graph = SourceGraph.Create(semanticModel);
        }

        public static SourceGraph ConstructASTGraph(SemanticModel semanticModel)
        {
            var visitor = new ASTGraphExtractor(semanticModel);
            visitor.Visit(semanticModel.SyntaxTree.GetRoot());
            return visitor._graph;
        }

        private void AddToken(SyntaxNode parentNode, SyntaxToken token)
        {
            if (_lastAddedToken.HasValue)
            {
                _graph.AddEdge(_lastAddedToken.Value, SourceGraphEdge.NextToken, token);
            }
            _lastAddedToken = token;
        }

        public override void Visit(SyntaxNode node)
        {
            foreach (var child in node.ChildNodesAndTokens())
            {
                _graph.AddEdge(node, SourceGraphEdge.Child, child);
                if (child.IsNode)
                {
                    Visit(child.AsNode());
                }
                else
                {
                    AddToken(node, child.AsToken());
                }
            }
        }
    }

    public class MethodInfoGraphExtractor : CSharpSyntaxWalker
    {
        private readonly DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> _graph;
        private readonly SemanticModel _semanticModel;
        private readonly Stack<SyntaxNodeOrToken> _returningPoint = new Stack<SyntaxNodeOrToken>();
        private readonly Dictionary<IMethodSymbol, HashSet<SyntaxNodeOrToken>> _boundMethods = new Dictionary<IMethodSymbol, HashSet<SyntaxNodeOrToken>>();

        private MethodInfoGraphExtractor(SemanticModel semanticModel, DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> graph)
        {
            _graph = graph;
            _semanticModel = semanticModel;
        }

        public static void ConstructMethodInfoGraph(SemanticModel semanticModel,
            DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> graph)
        {
            var extractor = new MethodInfoGraphExtractor(semanticModel, graph);
            extractor.Visit(semanticModel.SyntaxTree.GetRoot());

            // For all methods that are called at least twice, bind them with a dummy symbol
            foreach (var (invocationSymbol, invocationSites) in extractor._boundMethods.Where(kv => kv.Value.Count > 1))
            {
                var dummySymbol = SyntaxFactory.Identifier(invocationSymbol.Name);
                foreach (var invocation in invocationSites)
                {
                    graph.AddEdge(invocation, SourceGraphEdge.BindsToSymbol, dummySymbol);
                }
            }
        }
        
        public override void VisitReturnStatement(ReturnStatementSyntax node)
        {
            Debug.Assert(_returningPoint.Count > 0);
            if (node.Expression != null)
            {
                _graph.AddEdge(node.Expression, SourceGraphEdge.ReturnsTo, _returningPoint.Peek());
            }
            base.VisitReturnStatement(node);
        }

        public override void VisitInvocationExpression(InvocationExpressionSyntax node)
        {
            base.VisitInvocationExpression(node);

            var invocationSymbol = (IMethodSymbol)_semanticModel.GetSymbolInfo(node).Symbol;

            if (invocationSymbol == null) return;

            if (!_boundMethods.TryGetValue(invocationSymbol, out var boundInvocations))
            {
                boundInvocations = new HashSet<SyntaxNodeOrToken>();
                _boundMethods.Add(invocationSymbol, boundInvocations);
            }
            boundInvocations.Add(node);

            foreach (var arg in node.ArgumentList.Arguments)
            {
                var paramSymbol = DetermineParameter(node.ArgumentList, arg, invocationSymbol);
                Debug.Assert(arg.Expression != null);
                if (paramSymbol == null) continue; // rarely happens on __arglist (probably present only on coreclr)
                
                SyntaxNodeOrToken dummySymbol = SyntaxFactory.Identifier(paramSymbol.Name);
                
                // Create a dummy node for the formal parameter symbol
                _graph.AddEdge(arg.Expression, SourceGraphEdge.FormalArgName, dummySymbol);
            }
        }

        public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitMethodDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitConstructorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitDestructorDeclaration(DestructorDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitDestructorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitConversionOperatorDeclaration(ConversionOperatorDeclarationSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitConversionOperatorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitOperatorDeclaration(OperatorDeclarationSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitOperatorDeclaration(node);
            _returningPoint.Pop();
        }


        public override void VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitPropertyDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitIndexerDeclaration(IndexerDeclarationSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitIndexerDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitEventDeclaration(EventDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitEventDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitYieldStatement(YieldStatementSyntax node)
        {
            bool isReturnStatement = node.IsKind(SyntaxKind.YieldReturnStatement);
            if (isReturnStatement && node.Expression != null)
            {
                _graph.AddEdge(node.Expression, SourceGraphEdge.ReturnsTo, _returningPoint.Peek());
            } 
            base.VisitYieldStatement(node);
        }

        public override void VisitLocalFunctionStatement(LocalFunctionStatementSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitLocalFunctionStatement(node);
            _returningPoint.Pop();
        }

        public override void VisitSimpleLambdaExpression(SimpleLambdaExpressionSyntax node)
        {
            _returningPoint.Push(node); // Lambdas don't have names. Use the AST node instead
            base.VisitSimpleLambdaExpression(node);
            _returningPoint.Pop();
        }

        public override void VisitParenthesizedLambdaExpression(ParenthesizedLambdaExpressionSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitParenthesizedLambdaExpression(node);
            _returningPoint.Pop();
        }

        public override void VisitAnonymousMethodExpression(AnonymousMethodExpressionSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitAnonymousMethodExpression(node);
            _returningPoint.Pop();
        }
        
        /// <summary>
        /// Copied from Roslyn source code. Determines the parameter for a given argument
        /// </summary>
        /// <param name="argumentList"></param>
        /// <param name="argument"></param>
        /// <param name="symbol"></param>
        /// <returns></returns>
        public IParameterSymbol DetermineParameter(BaseArgumentListSyntax argumentList, ArgumentSyntax argument, IMethodSymbol symbol)
        {
            var parameters = symbol.Parameters;

            // Handle named argument
            if (argument.NameColon != null && !argument.NameColon.IsMissing)
            {
                var name = argument.NameColon.Name.Identifier.ValueText;
                return parameters.FirstOrDefault(p => p.Name == name);
            }

            // Handle positional argument
            var index = argumentList.Arguments.IndexOf(argument);
            if (index < 0)
            {
                return null;
            }

            if (index < parameters.Length)
            {
                return parameters[index];
            }

            // Handle Params
            var lastParameter = parameters.LastOrDefault();
            if (lastParameter == null)
            {
                return null;
            }

            if (lastParameter.IsParams)
            {
                return lastParameter;
            }

            return null;
        }
    }
}
