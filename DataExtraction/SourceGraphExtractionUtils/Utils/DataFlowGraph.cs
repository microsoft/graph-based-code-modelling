using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace SourceGraphExtractionUtils.Utils
{
    internal class DataFlowGraphExtractor : CSharpSyntaxWalker
    {
        /// <summary>
        ///     Actual data flow graph.
        /// </summary>
        private readonly DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> _graph;

        private readonly SemanticModel _semanticModel;

        /// <summary>
        ///     Stack corresponding to lex. scopes, each element mapping variables in scope to their last use (target nodes are
        ///     IdentifierNameSyntax)
        /// </summary>
        private readonly Stack<Dictionary<string, HashSet<SyntaxNode>>> _varToLastUsesStack =
            new Stack<Dictionary<string, HashSet<SyntaxNode>>>();

        /// <summary>
        ///     Stack corresponding to lex. scopes, each element mapping variables in scope to their last write (target nodes are
        ///     nodes that set variables, e.g. assignments, foreach loops, parameters, ...)
        /// </summary>
        private readonly Stack<Dictionary<string, HashSet<SyntaxNode>>> _varToLastWritesStack =
            new Stack<Dictionary<string, HashSet<SyntaxNode>>>();

        private DataFlowGraphExtractor(SemanticModel semanticModel, DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> graph = null)
        {
            _graph = graph ?? new DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge>();
            _semanticModel = semanticModel;
        }

        public static DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> ConstructDataFlowGraph(SemanticModel semanticModel, DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge> graph = null)
        {
            var visitor = new DataFlowGraphExtractor(semanticModel, graph);
            visitor.Visit(semanticModel.SyntaxTree.GetRoot());
            return visitor._graph;
        }

        #region Internal helpers to update information-holding data structures

        private bool IsVariableLike(IdentifierNameSyntax node)
        {
            var nodeSymbol = _semanticModel.GetSymbolInfo(node).Symbol;
            return nodeSymbol is IParameterSymbol || nodeSymbol is ILocalSymbol || nodeSymbol is IFieldSymbol ||
                   nodeSymbol is IEventSymbol || nodeSymbol is IPropertySymbol;
        }

        private IEnumerable<Tuple<IdentifierNameSyntax, string>> GetAllKnownVariablesInSyntaxTree(SyntaxNode node)
        {
            foreach (var descendant in node.DescendantNodesAndSelf())
            {
                var descendantId = descendant as IdentifierNameSyntax;
                if (descendantId == null) continue;
                var idName = descendantId.Identifier.ValueText;
                if (IsVariableLike(descendantId) && _varToLastUsesStack.Peek().ContainsKey(idName))
                {
                    yield return Tuple.Create(descendantId, idName);
                }
            }
        }

        private void RecordFlow(IDictionary<string, HashSet<SyntaxNode>> context, SyntaxNode useNode, string varName,
            SourceGraphEdge edgeType, bool replaceContextInfo = true)
        {
            //Console.WriteLine("Recording {0} flow for variable {1} from {2}", edgeType, varName, useNode.ToString().Replace("\n", "\\n").Replace("\r", "\\r"));
            // First, connect this node to existing parents:
            foreach (var lastUse in context[varName])
            {
                //Console.WriteLine("  To {0}", lastUse.ToString().Replace("\n", "\\n").Replace("\r", "\\r"));
                _graph.AddEdge(useNode, edgeType, lastUse);
            }
            if (replaceContextInfo)
            {
                context[varName] = new HashSet<SyntaxNode> { useNode };
            }
        }

        private void RecordVariableComputation(SyntaxNode assignedVariable, SyntaxNode assignedExpression,
            IEnumerable<Tuple<IdentifierNameSyntax, string>> indexers = null)
        {
            //Record computation:
            var usedVars =
                GetAllKnownVariablesInSyntaxTree(assignedExpression)
                    .Concat(indexers ?? Enumerable.Empty<Tuple<IdentifierNameSyntax, string>>());
            foreach (var usedVar in usedVars)
            {
                _graph.AddEdge(assignedVariable, SourceGraphEdge.ComputedFrom, usedVar.Item1);
            }
            _graph.AddEdge(assignedVariable, SourceGraphEdge.ComputedFrom, assignedExpression);
        }

        private void RecordVariableDeclaration(SyntaxNode declarationNode, string varName, SyntaxNode initializer = null,
            bool isComputed = true)
        {
            // There are cases where a declaration has no name (or use), e.g. catch(IOException) {..}. These variables don't appear anywhere
            if (varName == null)
                return;

            //Console.WriteLine("Recording declaration of {0} at {1}", varName, declarationNode.ToString().Replace("\n", "\\n").Replace("\r", "\\r"));

            _varToLastUsesStack.Peek()[varName] = new HashSet<SyntaxNode> { declarationNode };
            if (initializer == null)
            {
                _varToLastWritesStack.Peek()[varName] = new HashSet<SyntaxNode>();
            }
            else
            {
                if (isComputed)
                {
                    RecordVariableComputation(declarationNode, initializer);
                }
                _varToLastWritesStack.Peek()[varName] = new HashSet<SyntaxNode> { declarationNode };
            }
        }

        private static Dictionary<TKey, HashSet<TValue>> DeepCloneTopContext<TKey, TValue>(
            Stack<Dictionary<TKey, HashSet<TValue>>> contextStack)
        {
            var topContext = contextStack.Peek();
            var clonedContext = new Dictionary<TKey, HashSet<TValue>>();
            foreach (var kv in topContext)
            {
                clonedContext[kv.Key] = new HashSet<TValue>(kv.Value);
            }
            contextStack.Push(clonedContext);

            return clonedContext;
        }

        private void HandleParallelBlocks(IEnumerable<Action> parallelBlockExecutors, bool maySkip = false)
        {
            var newLastUses = new List<Dictionary<string, HashSet<SyntaxNode>>>();
            var newLastWrites = new List<Dictionary<string, HashSet<SyntaxNode>>>();
            foreach (var blockExecutor in parallelBlockExecutors)
            {
                //Create fresh contexts, execute the blocks, and pop context again. Store newly created context to create new result later on...
                newLastUses.Add(DeepCloneTopContext(_varToLastUsesStack));
                newLastWrites.Add(DeepCloneTopContext(_varToLastWritesStack));

                blockExecutor.Invoke();

                _varToLastUsesStack.Pop();
                _varToLastWritesStack.Pop();
            }

            //Extract variables in scope out into fresh list to avoid bane of concurrent modification
            var varToLastUses = _varToLastUsesStack.Peek();
            var varsInScope = varToLastUses.Keys.ToList();
            var varToLastWrites = _varToLastWritesStack.Peek();

            //If skipping the parallel blocks completely is allowed (e.g., in loops), then also treat the existing context as a "new" one:
            if (maySkip)
            {
                newLastUses.Add(varToLastUses);
                newLastWrites.Add(varToLastWrites);
            }

            foreach (var variable in varsInScope)
            {
                varToLastUses[variable] =
                    new HashSet<SyntaxNode>(newLastUses.SelectMany(newLastUse => newLastUse[variable]));
                varToLastWrites[variable] =
                    new HashSet<SyntaxNode>(newLastWrites.SelectMany(newLastWrite => newLastWrite[variable]));
            }
        }

        #endregion

        #region Visitors for control-flow elements and statements

        public override void VisitBlock(BlockSyntax node)
        {
            HandleParallelBlocks(new Action[]
            {
                () =>
                {
                    foreach (var statement in node.Statements)
                    {
                        Visit(statement);
                    }
                }
            });
        }

        public override void VisitIfStatement(IfStatementSyntax node)
        {
            Visit(node.Condition);
            HandleParallelBlocks(new Action[]
            {
                () => Visit(node.Statement),
                () => Visit(node.Else)
            });
        }

        public override void VisitSwitchStatement(SwitchStatementSyntax node)
        {
            Visit(node.Expression);
            HandleParallelBlocks(node.Sections.Select<SwitchSectionSyntax, Action>(switchSection => () =>
            {
                foreach (var statement in switchSection.Statements)
                {
                    Visit(statement);
                }
            }));
        }

        public override void VisitForStatement(ForStatementSyntax node)
        {
            Visit(node.Declaration);
            Visit(node.Condition);
            HandleParallelBlocks(new Action[]
            {
                () =>
                {
                    Visit(node.Statement);
                    foreach (var incrementor in node.Incrementors)
                    {
                        Visit(incrementor);
                    }
                    Visit(node.Condition);
                    Visit(node.Statement);
                    foreach (var incrementor in node.Incrementors)
                    {
                        Visit(incrementor);
                    }
                }
            }, true);
        }

        public override void VisitForEachStatement(ForEachStatementSyntax node)
        {
            Visit(node.Expression);
            HandleParallelBlocks(new Action[]
            {
                () =>
                {
                    RecordVariableDeclaration(node, node.Identifier.ValueText, node.Expression);
                    Visit(node.Statement);
                    Visit(node.Expression);
                    Visit(node.Statement);
                }
            }, true);
        }

        public override void VisitForEachVariableStatement(ForEachVariableStatementSyntax node)
        {
            Visit(node.Expression);
            HandleParallelBlocks(new Action[]
            {
                () =>
                {
                    Visit(node.Variable);
                    Visit(node.Statement);
                    Visit(node.Expression);
                    Visit(node.Statement);
                }
            }, true);
        }

        public override void VisitWhileStatement(WhileStatementSyntax node)
        {
            Visit(node.Condition);
            HandleParallelBlocks(new Action[] {
                () =>
                {
                    Visit(node.Statement);
                    Visit(node.Condition);
                    Visit(node.Statement);
                }
            }, true);
        }

        public override void VisitDoStatement(DoStatementSyntax node)
        {
            Visit(node.Statement);
            Visit(node.Condition);
            HandleParallelBlocks(new Action[]
            {
                () =>
                {
                    Visit(node.Statement);
                    Visit(node.Condition);
                    Visit(node.Statement);
                }
            }, true);
        }

        public override void VisitTryStatement(TryStatementSyntax node)
        {
            var catchesHandlers =
                node.Catches.Select<CatchClauseSyntax, Action>(
                    catchClause => (() =>
                    {
                        var catchDeclaration = catchClause.Declaration;
                        if (catchDeclaration != null)  // you may use "catch { .. }" without a declaration
                        {
                            RecordVariableDeclaration(catchDeclaration, catchDeclaration.Identifier.ValueText,
                                catchDeclaration, false);
                        }
                        Visit(catchClause.Block);
                    }));
            HandleParallelBlocks(new Action[] { () => Visit(node.Block) }.Concat(catchesHandlers));
            Visit(node.Finally);
        }

        public override void VisitUsingStatement(UsingStatementSyntax node)
        {
            HandleParallelBlocks(new Action[]
            {
                () =>
                {
                    Visit(node.Declaration);
                    Visit(node.Statement);
                }
            });
        }

        public override void VisitParameter(ParameterSyntax node)
        {
            var varName = node.Identifier.ValueText;
            _varToLastWritesStack.Peek()[varName] = new HashSet<SyntaxNode> { node };
            _varToLastUsesStack.Peek()[varName] = new HashSet<SyntaxNode> { node };
        }

        public override void VisitVariableDeclaration(VariableDeclarationSyntax node)
        {
            foreach (var variable in node.Variables)
            {
                var varName = variable.Identifier.ValueText;
                RecordVariableDeclaration(variable, varName, variable.Initializer);
            }
        }

        public override void VisitSingleVariableDesignation(SingleVariableDesignationSyntax node)
        {
            RecordVariableDeclaration(node, node.Identifier.ValueText, null);
        }

        public override void VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            var propName = node.Identifier.ValueText;
            RecordVariableDeclaration(node, propName, node.Initializer);
            Visit(node.AccessorList);
        }

        public override void VisitDefaultExpression(DefaultExpressionSyntax node)
        {
            // do nothing
        }

        public override void VisitAssignmentExpression(AssignmentExpressionSyntax node)
        {
            //Process the full assignment (i.e., rhs, indices, ... of it):
            Visit(node.Left);
            Visit(node.Right);

            //Get variables used in lhs of assignment:
            var leftVars = GetAllKnownVariablesInSyntaxTree(node.Left).ToList();
            if (leftVars.Count == 0)
            {
                Console.WriteLine("Could not determine written variable in assignment '{0}'", node);
                return;
            }
            var writtenVar = leftVars.First();
            var indexers = leftVars.Skip(1);

            //Record value computation:
            RecordVariableComputation(writtenVar.Item1, node.Right, indexers);

            //Record write to variable (note: this has to happen after recording the variable computation, to handle things like cur = cur.Parent correctly):
            RecordFlow(_varToLastWritesStack.Peek(), writtenVar.Item1, writtenVar.Item2, SourceGraphEdge.LastWrite);
            _varToLastUsesStack.Peek()[writtenVar.Item2] = new HashSet<SyntaxNode> { writtenVar.Item1 };
        }

        public override void VisitPostfixUnaryExpression(PostfixUnaryExpressionSyntax node)
        {
            base.VisitPostfixUnaryExpression(node);
            if (node.IsKind(SyntaxKind.PostDecrementExpression) || node.IsKind(SyntaxKind.PostIncrementExpression))
            {
                var variables = GetAllKnownVariablesInSyntaxTree(node.Operand).ToList();
                if (variables.Count == 0)
                {
                    Console.WriteLine("Could not determine written variable in assignment '{0}'", node);
                    return;
                }
                var writtenVar = variables.First();
                RecordFlow(_varToLastWritesStack.Peek(), writtenVar.Item1, writtenVar.Item2, SourceGraphEdge.LastWrite);
            }
        }

        public override void VisitPrefixUnaryExpression(PrefixUnaryExpressionSyntax node)
        {
            if (node.IsKind(SyntaxKind.PreDecrementExpression) || node.IsKind(SyntaxKind.PreIncrementExpression))
            {
                var variables = GetAllKnownVariablesInSyntaxTree(node.Operand).ToList();
                if (variables.Count == 0)
                {
                    Console.WriteLine("Could not determine written variable in assignment '{0}'", node);
                    return;
                }
                var writtenVar = variables.First();
                RecordFlow(_varToLastWritesStack.Peek(), writtenVar.Item1, writtenVar.Item2, SourceGraphEdge.LastWrite);
            }
            base.VisitPrefixUnaryExpression(node);
        }

        public override void VisitIdentifierName(IdentifierNameSyntax node)
        {
            if (IsVariableLike(node))
            {
                var usedId = node.Identifier.ValueText;
                if (_varToLastUsesStack.Peek().ContainsKey(usedId))
                {
                    RecordFlow(_varToLastUsesStack.Peek(), node, usedId, SourceGraphEdge.LastUse);
                    RecordFlow(_varToLastWritesStack.Peek(), node, usedId, SourceGraphEdge.LastWrite, false);
                }
                else
                {
                    var parent = node.Parent;
                    Console.WriteLine("Unknown id {0} in syntax {1}: {2}", usedId, parent.GetType().FullName,
                        node.Parent.ToString().Replace("\n", "\\n").Replace("\r", "\\r"));
                    Console.WriteLine("  at ({0})", node.SyntaxTree.GetLineSpan(node.Span));
                }
            }
        }

        public override void VisitMemberAccessExpression(MemberAccessExpressionSyntax node)
        {
            //Special-case handling of accesses to "this"
            var baseIdSyntax = node.Expression as IdentifierNameSyntax;
            if (baseIdSyntax != null && baseIdSyntax.Identifier.ValueText == "this" || node.Expression is ThisExpressionSyntax)
            {
                Visit(node.Name);
            }
            //For the rest, we just use the left-hand side of the member access:
            else
            {
                Visit(node.Expression);
            }
        }

        public override void VisitInitializerExpression(InitializerExpressionSyntax node)
        {
            //Special-case object initializers, which look like assignments and thus confuse us:
            if (node.Kind() == SyntaxKind.ObjectInitializerExpression)
            {
                foreach (var initializer in node.Expressions)
                {
                    var initializerAssignment = initializer as AssignmentExpressionSyntax;
                    Visit(initializerAssignment != null ? initializerAssignment.Right : initializer);
                }
            }
            else
            {
                base.VisitInitializerExpression(node);
            }
        }

        #endregion

        #region Visitors for non-(statement|expression) things

        //We can abstract away from different kinds of type declarations / member declarations:
        private void VisitTypeDeclaration(TypeDeclarationSyntax node)
        {
            _varToLastUsesStack.Push(new Dictionary<string, HashSet<SyntaxNode>>());
            _varToLastWritesStack.Push(new Dictionary<string, HashSet<SyntaxNode>>());

            //As C# isn't being picky about order of declarations, we re-order things here. First we declare fields, properties, then we run all constructors in parallel, then we do all the others.
            var fieldMemberIndices = new HashSet<int>();
            var propertyMemberIndices = new HashSet<int>();
            var constructorMemberIndices = new HashSet<int>();
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
            }
            foreach (var fieldDecl in fieldMemberIndices.Select(i => node.Members[i]))
            {
                Visit(fieldDecl);
            }
            foreach (var propertyDecl in propertyMemberIndices.Select(i => node.Members[i]))
            {
                Visit(propertyDecl);
            }

            var constructorHandlers = constructorMemberIndices.Select<int, Action>(i => (() => Visit(node.Members[i])));
            HandleParallelBlocks(constructorHandlers, maySkip: constructorMemberIndices.Count == 0);

            for (int i = 0; i < node.Members.Count; ++i)
            {
                if (!(fieldMemberIndices.Contains(i) || propertyMemberIndices.Contains(i) || constructorMemberIndices.Contains(i)))
                {
                    Visit(node.Members[i]);
                }
            }

            _varToLastUsesStack.Pop();
            _varToLastWritesStack.Pop();
        }

        private void VisitBaseMethodDeclaration(BaseMethodDeclarationSyntax node)
        {
            DeepCloneTopContext(_varToLastUsesStack);
            DeepCloneTopContext(_varToLastWritesStack);
            foreach (var parameter in node.ParameterList.Parameters)
            {
                Visit(parameter);
            }
            //Declare "this" in scope if we are non-static:
            if (node.Modifiers.All(modifierToken => modifierToken.Kind() != SyntaxKind.StaticKeyword))
            {
                RecordVariableDeclaration(node, "this", node, false);
            }
            Visit(node.Body);

            _varToLastUsesStack.Pop();
            _varToLastWritesStack.Pop();
        }

        public override void VisitClassDeclaration(ClassDeclarationSyntax node)
            => VisitTypeDeclaration(node);

        public override void VisitInterfaceDeclaration(InterfaceDeclarationSyntax node)
            => VisitTypeDeclaration(node);

        public override void VisitStructDeclaration(StructDeclarationSyntax node)
            => VisitTypeDeclaration(node);

        public override void VisitConversionOperatorDeclaration(ConversionOperatorDeclarationSyntax node)
            => VisitBaseMethodDeclaration(node);

        public override void VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
            => VisitBaseMethodDeclaration(node);

        public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
            => VisitBaseMethodDeclaration(node);

        public override void VisitOperatorDeclaration(OperatorDeclarationSyntax node)
            => VisitBaseMethodDeclaration(node);

        public override void VisitDestructorDeclaration(DestructorDeclarationSyntax node)
            => VisitBaseMethodDeclaration(node);

        public override void VisitAccessorDeclaration(AccessorDeclarationSyntax node)
        {
            //In setters, the implicit variable "value" is introduced:
            if (node.Body != null && node.Keyword.Kind() == SyntaxKind.SetKeyword)
            {
                DeepCloneTopContext(_varToLastUsesStack);
                DeepCloneTopContext(_varToLastWritesStack);

                RecordVariableDeclaration(node, "value", node, false);
                Visit(node.Body);

                _varToLastUsesStack.Pop();
                _varToLastWritesStack.Pop();
            }
            else
            {
                base.VisitAccessorDeclaration(node);
            }
        }


        public override void VisitCompilationUnit(CompilationUnitSyntax node)
        {
            //Skip using / ... and stuff:
            foreach (var member in node.Members)
            {
                Visit(member);
            }
        }

        public override void VisitNamespaceDeclaration(NamespaceDeclarationSyntax node)
        {
            //Skip namespace name, and all defined delegates:
            foreach (var member in node.Members.Where(member => member.Kind() != SyntaxKind.DelegateDeclaration))
            {
                Visit(member);
            }
        }

        public override void VisitAttributeList(AttributeListSyntax node)
        {
            //Don't recurse into these.
        }

        public override void VisitEnumDeclaration(EnumDeclarationSyntax node)
        {
            //Don't recurse into these.
        }

        #endregion

        #region Test / debug helpers

        // ReSharper disable once UnusedMember.Local
        private void PrintContext(string locationName)
        {
            Console.WriteLine("#### Current context at {0}:", locationName);
            foreach (var kv in _varToLastUsesStack.Peek())
            {
                var varName = kv.Key;
                var lastUses = kv.Value;
                var lastWrites = _varToLastWritesStack.Peek()[varName];
                Console.WriteLine(" For variable {0}:", varName);
                foreach (var lastUseNode in lastUses)
                {
                    Console.WriteLine("  Last use: [Type {0}] {1}", lastUseNode.GetType().Name,
                        lastUseNode.SyntaxTree.GetLineSpan(lastUseNode.Span));
                }
                foreach (var lastUseNode in lastWrites)
                {
                    Console.WriteLine("  Last write: [Type {0}] {1}", lastUseNode.GetType().Name,
                        lastUseNode.SyntaxTree.GetLineSpan(lastUseNode.Span));
                }
            }
        }

        private static bool IsDeclaringNode(SyntaxNode node)
            =>
                node is FieldDeclarationSyntax || node is VariableDeclaratorSyntax || node is PropertyDeclarationSyntax ||
                node is AccessorDeclarationSyntax || node is VariableDesignationSyntax ||
                node is ParameterSyntax || node is ForEachStatementSyntax || node is CatchDeclarationSyntax ||
                node is UsingStatementSyntax || node is LockStatementSyntax;

        private void CheckUseRoots(SyntaxNode node, bool printTrace = false, string prefix = "")
        {
            if (printTrace)
            {
                var useType = IsDeclaringNode(node) ? "Declaration" : "Use";
                Console.WriteLine("{0}{1}: {2} at {3}", prefix, useType, node, node.SyntaxTree.GetLineSpan(node.Span));
            }

            var lastUseEdges = _graph.GetOutEdges(node).Where(edge => edge.Label == SourceGraphEdge.LastUse).ToList();
            if (lastUseEdges.Count == 0)
            {
                if (IsDeclaringNode(node))
                {
                    //Everything is fine
                }
                else
                {
                    Console.WriteLine("Non-Declaration has no outgoing edges: [Type {0}] {1}\n  at {2}",
                        node.GetType().FullName, node, node.SyntaxTree.GetLineSpan(node.Span));
                }
            }
            else
            {
                if (IsDeclaringNode(node))
                {
                    Console.WriteLine("Declaration has outgoing use edges: [Type {0}] {1}\n  at {2}",
                        node.GetType().FullName, node, node.SyntaxTree.GetLineSpan(node.Span));
                }
                foreach (var lastUseEdge in lastUseEdges)
                {
                    CheckUseRoots(lastUseEdge.Target.AsNode(), printTrace, prefix + "  ");
                }
            }
        }

        private void CheckWriteRoots(SyntaxNode node, bool printTrace = false, string prefix = null,
            SourceGraphEdge? inEdgeType = null)
        {
            if (printTrace)
            {
                //We also use prefix as a flag if we want to trace this specific write chain; and only trace chains starting from a non-declaration:
                if (!IsDeclaringNode(node) && prefix == null)
                {
                    Console.WriteLine("Use: {0} at {1}", node, node.SyntaxTree.GetLineSpan(node.Span));
                    prefix = "";
                }
                else if (prefix != null)
                {
                    Console.WriteLine("{0}{1}: {2} at {3}: {4}", prefix, inEdgeType, node,
                        node.SyntaxTree.GetLineSpan(node.Span), node);
                }
            }

            var computationEdges =
                _graph.GetOutEdges(node)
                    .Where(edge => edge.Label == SourceGraphEdge.LastWrite || edge.Label == SourceGraphEdge.ComputedFrom)
                    .ToList();
            //If we have no prior writes, that's fine (could be declared without initial value)
            foreach (var computationEdge in computationEdges)
            {
                var target = computationEdge.Target;
                if (!(target.AsNode() is AssignmentExpressionSyntax || IsDeclaringNode(target.AsNode())))
                {
                    Console.WriteLine(
                        $"LastWrite/ComputedFrom edge does not lead to assignment.\n  Current node: [Type {0}] {1}\n    at {2}\n  Target node: [Type {3}] {4}\n    at {5}",
                        node.GetType().FullName, node, node.SyntaxTree.GetLineSpan(node.Span),
                        target.GetType().FullName, target, target.SyntaxTree.GetLineSpan(target.Span));
                    Console.WriteLine("Oh no, write edge is not leading to write !!1");
                }
                CheckWriteRoots(target.AsNode(), printTrace, (prefix == null ? null : prefix + "  "), computationEdge.Label);
            }
        }

        public void CheckGraph()
        {
            foreach (var node in _graph.Nodes)
            {
                //Check that all identifier nodes can be traced back to a declaration, and that all write edges link to writes:
                if (node.AsNode() is IdentifierNameSyntax)
                {
                    CheckUseRoots(node.AsNode());
                }
                CheckWriteRoots(node.AsNode());
            }
        }

        #endregion
    }
}