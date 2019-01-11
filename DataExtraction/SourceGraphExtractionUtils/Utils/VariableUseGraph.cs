using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using DataExtraction.Utils;

namespace SourceGraphExtractionUtils.Utils
{
    public static class DataFlowGraphHelper
    {
        public static void AddDataFlowEdges(SourceGraph sourceGraph, SyntaxNodeOrToken tokenOfInterest, ICollection<SyntaxNodeOrToken> forbiddenNodes = null, ICollection<Edge<SyntaxNodeOrToken, SourceGraphEdge>> addedEdges = null)
        {

            //There's nothing before the declaration, so we don't need to bother:
            if (sourceGraph.DeclarationNodes.Contains(tokenOfInterest))
            {
                return;
            }

            //We only ever need to visit each node once, so collect visited nodes here:
            var visitedNodes = new HashSet<(SyntaxNodeOrToken, bool)>();

            //Start from all predecessors of the token of interest:
            var toVisit = new Stack<(SyntaxNodeOrToken node, bool haveFoundUse)>();
            foreach (var (_, label, target) in sourceGraph.GetOutEdges(tokenOfInterest))
            {
                if (label != SourceGraphEdge.LastUsedVariable || (forbiddenNodes?.Contains(target) ?? false))
                {
                    continue;
                }
                if (visitedNodes.Add((target, false)))
                {
                    toVisit.Push((target, false));
                }
            }

            string nodeLabelToLookFor = tokenOfInterest.ToString();
            while (toVisit.Count > 0)
            {
                var (node, haveFoundUse) = toVisit.Pop();
                if (node.ToString().Equals(nodeLabelToLookFor))
                {
                    if (!haveFoundUse)
                    {
                        var lastUseEdge = new Edge<SyntaxNodeOrToken, SourceGraphEdge>(tokenOfInterest, SourceGraphEdge.LastUse, node);
                        if (sourceGraph.AddEdge(lastUseEdge))
                        {
                            addedEdges?.Add(lastUseEdge);
                        }
                        haveFoundUse = true;
                    }

                    if (sourceGraph.WrittenVariableNodes.Contains(node))
                    {
                        var lastWriteEdge = new Edge<SyntaxNodeOrToken, SourceGraphEdge>(tokenOfInterest, SourceGraphEdge.LastWrite, node);
                        if (sourceGraph.AddEdge(lastWriteEdge))
                        {
                            addedEdges?.Add(lastWriteEdge);
                        }
                        //We are done with this path -- we found a use and a write!
                        continue;
                    }

                    //There's nothing before the declaration, so we don't need to bother to recurse further:
                    if (sourceGraph.DeclarationNodes.Contains(node))
                    {
                        continue;
                    }
                }

                foreach (var (_, label, target) in sourceGraph.GetOutEdges(node))
                {
                    if (label != SourceGraphEdge.LastUsedVariable || (forbiddenNodes?.Contains(target) ?? false))
                    {
                        continue;
                    }
                    if (visitedNodes.Add((target, haveFoundUse)))
                    {
                        toVisit.Push((target, haveFoundUse));
                    }
                }
            }
        }

        public static void AddDataFlowEdges(SourceGraph sourceGraph, IEnumerable<SyntaxNodeOrToken> tokensOfInterest)
        {
            foreach (var tokenOfInterest in tokensOfInterest)
            {
                AddDataFlowEdges(sourceGraph, tokenOfInterest);
            }
        }
    }

    public class VariableUseGraphExtractor : CSharpSyntaxVisitor<(SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes)>
    {
        /// <summary>
        ///     Actual variable use graph: Here, each identifier has an edge to every identifier that may have been used directly before it.
        ///     For example, consider this code:
        ///          x1 = 1;
        ///          x4 = x3 - x2
        ///     Here, we would have edges (x4, x2), (x2, x3), (x3, x1), reflecting the order in which values are consumed in the execution.
        /// </summary>
        private readonly SourceGraph _graph;

        private readonly SemanticModel _semanticModel;

        private VariableUseGraphExtractor(SemanticModel semanticModel, SourceGraph graph)
        {
            _graph = graph;
            _semanticModel = semanticModel;
        }

        public static void ConstructVariableUseGraph(SemanticModel semanticModel, SourceGraph graph)
        {
            new VariableUseGraphExtractor(semanticModel, graph).Visit(semanticModel.SyntaxTree.GetRoot());
        }

        #region Internal helpers
        public bool AddEdges(SyntaxToken source, IEnumerable<SyntaxToken> targets)
        {
            _graph.UsedVariableNodes.Add(source);
            bool addedEdge = false;
            foreach (var targetNode in targets)
            {
                addedEdge = _graph.AddEdge(source, SourceGraphEdge.LastUsedVariable, targetNode) || addedEdge;
            }
            return addedEdge;
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) Visit(SyntaxNode node)
        {
            return base.Visit(node);
        }

        private (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) HandleSequentialNodes(IEnumerable<SyntaxNode> nodes)
        {
            // Default behaviour is to assume that execution proceeds in order of nodes, and that we just have to chain things through.
            // We overwrite (hopefully) all cases in which that does not hold and which thus require special attention.
            SyntaxToken? firstEntryNode = null;
            IEnumerable<SyntaxToken> lastExitNodes = Enumerable.Empty<SyntaxToken>();

            foreach (var curNode in nodes)
            {
                if (curNode == null) continue;

                var (curEntryNode, curExitNodes) = Visit(curNode);

                if (firstEntryNode == null)
                {
                    firstEntryNode = curEntryNode;
                }

                // If this is a statement through which control flows, connect it to the rest, and continue with its exits:
                if (curEntryNode.HasValue)
                {
                    AddEdges(curEntryNode.Value, lastExitNodes);
                    lastExitNodes = curExitNodes;
                }
            }

            return (firstEntryNode, lastExitNodes);
        }

        private IEnumerable<SyntaxToken> HandleParallelNodes(IEnumerable<SyntaxToken> precedingNodes, IEnumerable<SyntaxNode> nodes)
        {
            List<SyntaxToken> allExitNodes = new List<SyntaxToken>();

            foreach (var curNode in nodes)
            {
                if (curNode == null) continue;

                var (curEntryNode, curExitNodes) = Visit(curNode);

                if (curEntryNode.HasValue)
                {
                    AddEdges(curEntryNode.Value, precedingNodes);
                }

                allExitNodes.AddRange(curExitNodes);
            }

            if (!allExitNodes.Any())
            {
                return precedingNodes;
            }

            return allExitNodes;
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) DefaultVisit(SyntaxNode node)
        {
            if (node != null)
            {
                return HandleSequentialNodes(node.ChildNodes());
            }
            else
            {
                return (null, Enumerable.Empty<SyntaxToken>());
            }
        }
        #endregion

        #region Visitors for control-flow elements
        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitBlock(BlockSyntax node)
            => HandleSequentialNodes(node.Statements);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitIfStatement(IfStatementSyntax node)
        {
            var (conditionEntry, conditionExits) = Visit(node.Condition);
            var thenElseExits = HandleParallelNodes(conditionExits, new SyntaxNode[] { node.Statement, node.Else });
            return (conditionEntry, thenElseExits);
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitSwitchStatement(SwitchStatementSyntax node)
        {
            var (conditionEntry, conditionExits) = Visit(node.Expression);
            var switchExits = HandleParallelNodes(conditionExits, node.Sections);
            return (conditionEntry, switchExits);
        }

        private (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) ChainOptionalResults(IEnumerable<(SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes)> data)
        {
            SyntaxToken? firstEntry = null;
            IEnumerable<SyntaxToken> curExits = Enumerable.Empty<SyntaxToken>();
            foreach (var pair in data)
            {
                if (pair.entryNode.HasValue)
                {
                    firstEntry = firstEntry ?? pair.entryNode;
                    AddEdges(pair.entryNode.Value, curExits);
                    curExits = pair.exitNodes;
                }
            }
            return (firstEntry, curExits);
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitForStatement(ForStatementSyntax node)
        {
            var declarationRes = Visit(node.Declaration);
            var incrementorRes = HandleSequentialNodes(node.Incrementors);
            var conditionRes = Visit(node.Condition);
            var bodyRes = Visit(node.Statement);
            // Control flows from declaration to condition to body to incrementors to condition:
            //TODO: This should actually also handle things such as "break" (which doesn't pass through the condition):
            return ChainOptionalResults(new[] { declarationRes, conditionRes, bodyRes, incrementorRes, conditionRes });
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitForEachStatement(ForEachStatementSyntax node)
        {
            var declarationRes = HandleDeclaration(node.Identifier, node.Expression);
            var bodyRes = Visit(node.Statement);

            // Control flows from declaration to body to declaration:
            //TODO: This should actually also handle things such as "break" (which doesn't pass through the condition):
            return ChainOptionalResults(new[] { declarationRes, bodyRes, declarationRes });
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitForEachVariableStatement(ForEachVariableStatementSyntax node)
        {
            var declarationRes = Visit(node.Expression);
            var variableRes = Visit(node.Variable);
            var bodyRes = Visit(node.Statement);

            // The foreach is an implicit assignment to its variable:
            if (node.Variable is DeclarationExpressionSyntax varDeclaration)
            {
                HandleTupleAssignment(varDeclaration.Designation, node.Expression, isDeclaration: true);
            }
            // Control flows from declaration to body to declaration:
            //TODO: This should actually also handle things such as "break" (which doesn't pass through the condition):
            return ChainOptionalResults(new[] { declarationRes, variableRes, bodyRes, declarationRes });
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitWhileStatement(WhileStatementSyntax node)
        {
            var conditionRes = Visit(node.Condition);
            var bodyRes = Visit(node.Statement);
            // Control flows from condition to body to condition:
            //TODO: This should actually also handle things such as "break" (which doesn't pass through the condition):
            return ChainOptionalResults(new[] { conditionRes, bodyRes, conditionRes });
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitDoStatement(DoStatementSyntax node)
        {
            var conditionRes = Visit(node.Condition);
            var bodyRes = Visit(node.Statement);
            // Control flows from body to condition to body to condition:
            //TODO: This should actually also handle things such as "break" (which doesn't pass through the condition):
            return ChainOptionalResults(new[] { bodyRes, conditionRes, bodyRes, conditionRes });
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitTryStatement(TryStatementSyntax node)
        {
            var (bodyEntry, bodyExits) = Visit(node.Block);
            var (finallyEntry, finallyExits) = Visit(node.Finally);
            //TODO: This is imprecise in that this assumes that the body runs to the end...
            var catchExits = HandleParallelNodes(bodyExits, node.Catches);

            var bodyCatchExits = bodyExits.Concat(catchExits);

            if (finallyEntry.HasValue)
            {
                AddEdges(finallyEntry.Value, bodyCatchExits);
                return (bodyEntry, finallyExits);
            }
            else
            {
                return (bodyEntry, bodyCatchExits);
            }
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitUsingStatement(UsingStatementSyntax node)
        {
            var declarationRes = Visit(node.Declaration);
            var exprRes = Visit(node.Expression);
            var bodyRes = Visit(node.Statement);
            // Control flows from declaration to expression to body:
            return ChainOptionalResults(new[] { declarationRes, exprRes, bodyRes });
        }
        #endregion

        #region Visitors for statements and declarations
        private (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) HandleDeclaration(SyntaxToken declaredId, SyntaxNode initialValue)
        {
            _graph.UsedVariableNodes.Add(declaredId);
            _graph.DeclarationNodes.Add(declaredId);
            _graph.WrittenVariableNodes.Add(declaredId);
            if (initialValue != null)
            {
                var (initializerEntry, initializerExits) = Visit(initialValue);
                foreach (var usedToken in initialValue.DescendantTokens().Where(token => _graph.UsedVariableNodes.Contains(token)))
                {
                    _graph.AddEdge(declaredId, SourceGraphEdge.ComputedFrom, usedToken);
                }
                AddEdges(declaredId, initializerExits);
                if (initializerEntry == null) // Happens when initial value is constant
                {
                    return (declaredId, new[] { declaredId });
                }
                else
                {
                    return (initializerEntry, new[] { declaredId });
                }
            }
            else
            {
                return (declaredId, new[] { declaredId });
            }
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitParameter(ParameterSyntax node)
            => HandleDeclaration(node.Identifier, node.Default);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitSingleVariableDesignation(SingleVariableDesignationSyntax node)
            => HandleDeclaration(node.Identifier, null);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitVariableDeclaration(VariableDeclarationSyntax node)
            => HandleSequentialNodes(node.Variables);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitVariableDeclarator(VariableDeclaratorSyntax node)
            => HandleDeclaration(node.Identifier, node.Initializer);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            if (node.ExpressionBody == null && (node.AccessorList == null || node.AccessorList.Accessors.Count == 0))
            {
                return HandleDeclaration(node.Identifier, node.Initializer);
            }
            else if (node.ExpressionBody != null)
            {
                return HandleDeclaration(node.Identifier, node.ExpressionBody.Expression);
            }
            else
            {
                var getter = node.AccessorList.Accessors.Where(acc => acc.Keyword.IsKind(SyntaxKind.GetKeyword)).FirstOrDefault();
                var setter = node.AccessorList.Accessors.Where(acc => acc.Keyword.IsKind(SyntaxKind.SetKeyword)).FirstOrDefault();
                var (initializerEntryNode, initializerExitNodes) = HandleDeclaration(node.Identifier, (SyntaxNode)getter ?? node.Initializer);
                if (setter == null)
                {
                    return (initializerEntryNode, initializerExitNodes);
                }
                var (setterEntryNode, setterExitNodes) = Visit(setter);
                return ChainOptionalResults(new[] { (initializerEntryNode, initializerExitNodes), (setterEntryNode, setterExitNodes) });
            }
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitDefaultExpression(DefaultExpressionSyntax node)
            => (null, Enumerable.Empty<SyntaxToken>());

        private void HandleTupleAssignment(VariableDesignationSyntax lhsNode, SyntaxNode rhsNode, bool isDeclaration = false)
        {
            // If we recognise a tuple on both sides, try to match them up; otherwise connect each lhs element to all rhs elements:
            if (lhsNode is ParenthesizedVariableDesignationSyntax lhsParenDesignationNode && rhsNode is TupleExpressionSyntax tupleNode)
            {
                foreach (var (lhsElement, rhsElement) in Enumerable.Zip(lhsParenDesignationNode.Variables, tupleNode.Arguments, (l, r) => (l, r.Expression)))
                {
                    HandleTupleAssignment(lhsElement, rhsElement, isDeclaration);
                }
            }
            else
            {
                var rhsUsedVars = rhsNode.DescendantTokens().Where(token => _graph.UsedVariableNodes.Contains(token)).ToArray();
                foreach (var writtenToken in lhsNode.DescendantTokens().Where(token => _graph.WrittenVariableNodes.Contains(token)))
                {
                    if (isDeclaration)
                    {
                        _graph.DeclarationNodes.Add(writtenToken);
                    }
                    foreach (var rhsUsedVar in rhsUsedVars)
                    {
                        _graph.AddEdge(writtenToken, SourceGraphEdge.ComputedFrom, rhsUsedVar);
                    }
                }
            }
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitAssignmentExpression(AssignmentExpressionSyntax node)
        {
            var (rhsEntry, rhsExits) = Visit(node.Right);
            var (lhsEntry, lhsExits) = Visit(node.Left);

            // lhsEntry will be None for assignments to static class variables of other classes.
            if (lhsEntry.HasValue)
            {
                AddEdges(lhsEntry.Value, rhsExits);

                // We distinguish tuple assignments from others:
                if (node.Left is DeclarationExpressionSyntax lhsDeclaration)
                {
                    HandleTupleAssignment(lhsDeclaration.Designation, node.Right);
                }
                else
                {
                    var writtenToken = lhsExits.First();
                    foreach (var usedToken in node.Right.DescendantTokens().Where(token => _graph.UsedVariableNodes.Contains(token)))
                    {
                        _graph.AddEdge(writtenToken, SourceGraphEdge.ComputedFrom, usedToken);
                    }
                    _graph.WrittenVariableNodes.Add(writtenToken);
                }
            }
            else if (!(node.Parent is InitializerExpressionSyntax))
            {
                // ComputedFrom edges will be added to the object being initialized and
                // not within the initializer.
                _graph.AddEdge(node.Left, SourceGraphEdge.ComputedFrom, node.Right);
            }

            if (rhsEntry != null)  // rhsEntry is null if the rhs is just constants
            {
                if (lhsExits.Any())
                {
                    return (rhsEntry, lhsExits);
                }
                else
                {
                    return (rhsEntry, rhsExits);
                }
            }
            else
            {
                return (lhsEntry, lhsExits);
            }
        }
        
        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitPostfixUnaryExpression(PostfixUnaryExpressionSyntax node)
        {
            if (node.IsKind(SyntaxKind.PostDecrementExpression) || node.IsKind(SyntaxKind.PostIncrementExpression))
            {
                var changedToken = RoslynUtils.GetAllVariableSymbolsInSyntaxTree(node, _semanticModel).FirstOrDefault();
                _graph.WrittenVariableNodes.Add(changedToken);
            }
            return Visit(node.Operand);
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitPrefixUnaryExpression(PrefixUnaryExpressionSyntax node)
        {
            if (node.IsKind(SyntaxKind.PreDecrementExpression) || node.IsKind(SyntaxKind.PreIncrementExpression))
            {
                var changedToken = RoslynUtils.GetAllVariableSymbolsInSyntaxTree(node, _semanticModel).FirstOrDefault();
                _graph.WrittenVariableNodes.Add(changedToken);
            }
            return Visit(node.Operand);
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitArgument(ArgumentSyntax node)
        {
            // Ignore the argument labels:
            // For example foo(arg1: bar), "arg1" should not be visited.
            return Visit(node.Expression);
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitIdentifierName(IdentifierNameSyntax node)
        {
            if (RoslynUtils.IsVariableLike(node, _semanticModel, out var _))
            {
                _graph.UsedVariableNodes.Add(node.Identifier);
                return (node.Identifier, new[] { node.Identifier });
            }
            else
            {
                return (null, Enumerable.Empty<SyntaxToken>());
            }
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitMemberAccessExpression(MemberAccessExpressionSyntax node)
        {
            //Special-case handling of accesses to "this"
            var baseIdSyntax = node.Expression as IdentifierNameSyntax;
            if (baseIdSyntax != null && baseIdSyntax.Identifier.ValueText == "this" || node.Expression is ThisExpressionSyntax)
            {
                return Visit(node.Name);
            }
            //For the rest, we just use the left-hand side of the member access:
            else
            {
                return Visit(node.Expression);
            }
        }
        #endregion

        #region Visitors for non-(statement|expression) things
        //We can abstract away from different kinds of type declarations / member declarations:
        private (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitTypeDeclaration(TypeDeclarationSyntax node)
        {
            //As C# isn't being picky about order of declarations, we re-order things here. First we declare fields, properties, then we run all constructors in parallel, then we do all the others.
            var fieldMemberIndices = new List<int>();
            var propertyMemberIndices = new List<int>();
            var constructorMemberIndices = new List<int>();
            var otherMemberIndices = new List<int>();
            for (int i = 0; i < node.Members.Count; ++i)
            {
                var curMember = node.Members[i];
                if (curMember is FieldDeclarationSyntax)
                {
                    fieldMemberIndices.Add(i);
                }
                else if (curMember is PropertyDeclarationSyntax)
                {
                    propertyMemberIndices.Add(i);
                }
                else if (curMember is ConstructorDeclarationSyntax)
                {
                    constructorMemberIndices.Add(i);
                }
                else
                {
                    otherMemberIndices.Add(i);
                }
            }
            var (dataDeclarationEntry, dataDeclarationExits) = HandleSequentialNodes(fieldMemberIndices.Select(i => node.Members[i]).Concat(propertyMemberIndices.Select(i => node.Members[i])));

            var constructorExits = HandleParallelNodes(dataDeclarationExits, constructorMemberIndices.Select(i => node.Members[i]));

            var preludeExits = constructorExits.Any() ? constructorExits : dataDeclarationExits;

            var methodExits = HandleParallelNodes(preludeExits, otherMemberIndices.Select(i => node.Members[i]));

            return (dataDeclarationEntry, methodExits);
        }

        private (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) HandleBaseMethodDeclaration(BaseMethodDeclarationSyntax node)
        {
            SyntaxToken? bodyEntry;
            IEnumerable<SyntaxToken> bodyExits;
            if (node.Body != null)
            {
                (bodyEntry, bodyExits) = Visit(node.Body);
            }
            else if (node.ExpressionBody != null)
            {
                (bodyEntry, bodyExits) = Visit(node.ExpressionBody);
            }
            else
            {
                (bodyEntry, bodyExits) = (null, Enumerable.Empty<SyntaxToken>());
            }

            if (node.ParameterList.Parameters.Count > 0)
            {
                var (parameterEntry, parameterExits) = HandleSequentialNodes(node.ParameterList.Parameters);
                if (bodyEntry.HasValue)
                {
                    AddEdges(bodyEntry.Value, parameterExits);
                    return (parameterEntry, bodyExits);
                }
                else
                {
                    return (parameterEntry, parameterExits);
                }
            }
            else
            {
                return (bodyEntry, bodyExits);
            }
        }

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitClassDeclaration(ClassDeclarationSyntax node)
            => VisitTypeDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitInterfaceDeclaration(InterfaceDeclarationSyntax node)
            => VisitTypeDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitStructDeclaration(StructDeclarationSyntax node)
            => VisitTypeDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitConversionOperatorDeclaration(ConversionOperatorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitMethodDeclaration(MethodDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitOperatorDeclaration(OperatorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitDestructorDeclaration(DestructorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitCompilationUnit(CompilationUnitSyntax node)
            => (null, HandleParallelNodes(Enumerable.Empty<SyntaxToken>(), node.Members)); // Skips "using" / ... stuff:

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitNamespaceDeclaration(NamespaceDeclarationSyntax node)
            => (null, HandleParallelNodes(Enumerable.Empty<SyntaxToken>(), node.Members.Where(member => member.Kind() != SyntaxKind.DelegateDeclaration))); // Skips namespace name, and all defined delegates:

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitAttributeList(AttributeListSyntax node)
            => (null, Enumerable.Empty<SyntaxToken>());

        public override (SyntaxToken? entryNode, IEnumerable<SyntaxToken> exitNodes) VisitEnumDeclaration(EnumDeclarationSyntax node)
            => (null, Enumerable.Empty<SyntaxToken>());
        #endregion
    }
}