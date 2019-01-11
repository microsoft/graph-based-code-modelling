using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SourceGraphExtractionUtils.Utils
{
    internal class ExecutionPathGraph : CSharpSyntaxWalker
    {
        private readonly SemanticModel _semanticModel;
        private readonly SyntaxNode _root;

        private readonly Dictionary<SyntaxToken, int> _tokensOfInterestToLocations;


        private readonly Dictionary<int, HashSet<int>> _tokenPathParents = new Dictionary<int, HashSet<int>>();
        private HashSet<int> _currentPathParents = new HashSet<int>();
        private readonly Stack<HashSet<int>> _breakingFromCx = new Stack<HashSet<int>>();
        private readonly Stack<HashSet<int>> _continuingFromCx = new Stack<HashSet<int>>();
        private readonly Stack<HashSet<int>> _returningFromCx = new Stack<HashSet<int>>();

        public ExecutionPathGraph(SemanticModel semanticModel, List<SyntaxToken> tokens, HashSet<int> tokenIdsOfInterest) : base(SyntaxWalkerDepth.Token)
        {
            _semanticModel = semanticModel;
            _root = semanticModel.SyntaxTree.GetRoot();
            _tokensOfInterestToLocations = tokenIdsOfInterest.ToDictionary(idx => tokens[idx], idx => idx);

        }

        public Dictionary<int, HashSet<int>> ExtractPathDependencies()
        {
            // For all top-level type declarations
            Visit(_root);
            return _tokenPathParents;
        }

        private void AddTypeDeclaration(TypeDeclarationSyntax typeDeclaration)
        {
            // First visit static fields/methods
            var staticFields = typeDeclaration.Members.OfType<FieldDeclarationSyntax>().Where(f => f.Modifiers.Any(t => t.IsKind(SyntaxKind.StaticKeyword))).ToList();
            var staticProperties = typeDeclaration.Members.OfType<PropertyDeclarationSyntax>().Where(f => f.Modifiers.Any(t => t.IsKind(SyntaxKind.StaticKeyword))).ToList();
            var staticConstructors = typeDeclaration.Members.OfType<ConstructorDeclarationSyntax>().Where(f => f.Modifiers.Any(t => t.IsKind(SyntaxKind.StaticKeyword))).ToList();
            foreach (var node in staticFields.Cast<SyntaxNode>().Concat(staticProperties).Concat(staticConstructors))
            {
                Visit(node);
            }
            

            // Then visit non-static fields
            var nonStaticFields = typeDeclaration.Members.OfType<FieldDeclarationSyntax>().Where(f => !f.Modifiers.Any(t => t.IsKind(SyntaxKind.StaticKeyword))).ToList();
            foreach (var node in nonStaticFields.Cast<SyntaxNode>())
            {
                Visit(node);
            }

            // Then visit in parallel all constructors
            var nonStaticConstructors = typeDeclaration.Members.OfType<ConstructorDeclarationSyntax>().Where(f => !f.Modifiers.Any(t => t.IsKind(SyntaxKind.StaticKeyword))).Cast<SyntaxNode>().ToList();
            RecordNodesInParallel(nonStaticConstructors);
            
            var nonStaticProperties = typeDeclaration.Members.OfType<PropertyDeclarationSyntax>().Where(f => !f.Modifiers.Any(t => t.IsKind(SyntaxKind.StaticKeyword))).ToList();
            foreach (var node in nonStaticProperties.Cast<SyntaxNode>())
            {
                Visit(node);
            }

            // Then visit in parallel all non-constructor methods (we ignore the fact that static methods could be in another location.
            var allMethods = typeDeclaration.Members.OfType<BaseMethodDeclarationSyntax>().Where(m => !(m is ConstructorDeclarationSyntax) && !(m is DestructorDeclarationSyntax)).Cast<SyntaxNode>().ToList();
            //RecordNodesInParallel(allNonStaticMethods);
            RecordOptionalNodes(allMethods);  // This should allow methods to have more context about fields

            // Then visit the destructor
            var destructors = typeDeclaration.Members.OfType<DestructorDeclarationSyntax>();
            foreach (var node in destructors)
            {
                Visit(node);
            }
        }

        private void RecordNodesInParallel(List<SyntaxNode> parallelNodes)
        {
            var inputPath = _currentPathParents;
            var outputPaths = new HashSet<int>();
            foreach (var node in parallelNodes)
            {
                _currentPathParents = new HashSet<int>(inputPath);
                Visit(node);
                outputPaths.UnionWith(_currentPathParents);
            }
            _currentPathParents = outputPaths;
        }

        private void RecordOptionalNodes(List<SyntaxNode> parallelNodes)
        {
            foreach (var node in parallelNodes)
            {
                RecordNodesInParallel(new List<SyntaxNode> { node, (SyntaxNode)SyntaxFactory.EmptyStatement() });
            }
        }

        public override void VisitClassDeclaration(ClassDeclarationSyntax node) => AddTypeDeclaration(node);
        public override void VisitEnumDeclaration(EnumDeclarationSyntax node) {
            foreach (var member in node.Members) Visit(member);
        }

        public override void VisitStructDeclaration(StructDeclarationSyntax node) => AddTypeDeclaration(node);
        public override void VisitInterfaceDeclaration(InterfaceDeclarationSyntax node) => AddTypeDeclaration(node);


        public override void VisitToken(SyntaxToken token)
        {
            int tokLocationIdx;
            if (_tokensOfInterestToLocations.TryGetValue(token, out tokLocationIdx))
            {
                _tokenPathParents.Add(tokLocationIdx, _currentPathParents);
                _currentPathParents = new HashSet<int>() { tokLocationIdx };
            }
        }

        public override void VisitIfStatement(IfStatementSyntax node)
        {
            Visit(node.Condition);
            SyntaxNode elseNode = (node.Else == null) ? (SyntaxNode)SyntaxFactory.EmptyStatement() : node.Else;
            RecordNodesInParallel(new List<SyntaxNode>() { node.Statement, elseNode });
        }

        public override void VisitSwitchStatement(SwitchStatementSyntax node)
        {
            Visit(node.Expression);
            _breakingFromCx.Push(new HashSet<int>());

            var incomingPathParents = new HashSet<int>(_currentPathParents);

            foreach (var section in node.Sections)
            {
                _currentPathParents.UnionWith(incomingPathParents);
                Visit(section);
            }

            var breakedFrom = _breakingFromCx.Pop();
            _currentPathParents.UnionWith(breakedFrom);
        }

        public override void VisitTryStatement(TryStatementSyntax node)
        {
            _returningFromCx.Push(new HashSet<int>());
            Visit(node.Block);
            
            var allCatchesOrEmpty = node.Catches.Cast<SyntaxNode>().ToList();
            allCatchesOrEmpty.Add(SyntaxFactory.EmptyStatement());
            RecordNodesInParallel(allCatchesOrEmpty);

            var returnedFrom = _returningFromCx.Pop();
            _currentPathParents.UnionWith(returnedFrom);
            if (node.Finally != null)
            {
                Visit(node.Finally);
            }
        }

        public override void VisitWhileStatement(WhileStatementSyntax node)
        {
            Visit(node.Condition);
            _breakingFromCx.Push(new HashSet<int>(_currentPathParents));
            _continuingFromCx.Push(new HashSet<int>());

            Visit(node.Statement);

            var breakedFrom = _breakingFromCx.Pop();
            _currentPathParents.UnionWith(breakedFrom);
            var continuedFrom = _continuingFromCx.Pop();
            _currentPathParents.UnionWith(continuedFrom);
        }

        public override void VisitDoStatement(DoStatementSyntax node)
        {
            _continuingFromCx.Push(new HashSet<int>());
            _breakingFromCx.Push(new HashSet<int>());

            Visit(node.Statement);

            var continuedFrom = _continuingFromCx.Pop();
            _currentPathParents.UnionWith(continuedFrom);

            Visit(node.Condition);

            var breakedFrom = _breakingFromCx.Pop();
            _currentPathParents.UnionWith(breakedFrom);
        }

        public override void VisitForEachVariableStatement(ForEachVariableStatementSyntax node)
        {
            Visit(node.Expression);

            _breakingFromCx.Push(new HashSet<int>(_currentPathParents));
            _continuingFromCx.Push(new HashSet<int>());

            Visit(node.Statement);

            var continuedFrom = _continuingFromCx.Pop();
            _currentPathParents.UnionWith(continuedFrom);

            Visit(node.Variable);

            var breakedFrom = _breakingFromCx.Pop();
            _currentPathParents.UnionWith(breakedFrom);
        }

        public override void VisitForEachStatement(ForEachStatementSyntax node)
        {
            Visit(node.Expression);
            VisitToken(node.Identifier);

            _breakingFromCx.Push(new HashSet<int>(_currentPathParents));
            _continuingFromCx.Push(new HashSet<int>());

            Visit(node.Statement);

            var breakedFrom = _breakingFromCx.Pop();
            _currentPathParents.UnionWith(breakedFrom);
            var continuedFrom = _continuingFromCx.Pop();
            _currentPathParents.UnionWith(continuedFrom);
        }

        public override void VisitForStatement(ForStatementSyntax node)
        {
            Visit(node.Declaration);

            foreach (var initializer in node.Initializers)
            {
                Visit(initializer);
            }

            Visit(node.Condition);

            _breakingFromCx.Push(new HashSet<int>(_currentPathParents));
            _continuingFromCx.Push(new HashSet<int>());

            Visit(node.Statement);


            var continuedFrom = _continuingFromCx.Pop();
            _currentPathParents.UnionWith(continuedFrom);

            foreach (var incremenentor in node.Incrementors) Visit(incremenentor);

            var breakedFrom = _breakingFromCx.Pop();
            _currentPathParents.UnionWith(breakedFrom);
        }

        public override void VisitBreakStatement(BreakStatementSyntax node)
        {
            _breakingFromCx.Peek().UnionWith(_currentPathParents);
            _currentPathParents = new HashSet<int>();
        }

        public override void VisitContinueStatement(ContinueStatementSyntax node)
        {
            _continuingFromCx.Peek().UnionWith(_currentPathParents);
            _currentPathParents = new HashSet<int>();
        }

        public void VisitBaseMethodDeclaration(BaseMethodDeclarationSyntax node)
        {            
            foreach (var param in node.ParameterList.Parameters) Visit(param);
            if (node is ConstructorDeclarationSyntax)
            {
                var constructor = node as ConstructorDeclarationSyntax;
                Visit(constructor.Initializer);
            }
            _returningFromCx.Push(new HashSet<int>());
            Visit(node.Body);
            Visit(node.ExpressionBody);
            _currentPathParents.UnionWith(_returningFromCx.Pop());
        }

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

        public override void VisitAnonymousMethodExpression(AnonymousMethodExpressionSyntax node)
        {
            if (node.ParameterList != null)
            {
                foreach (var param in node.ParameterList.Parameters)
                    Visit(param);
            }
            
            _returningFromCx.Push(new HashSet<int>());
            Visit(node.Body);
            _currentPathParents.UnionWith(_returningFromCx.Pop());
        }

        public override void VisitParenthesizedLambdaExpression(ParenthesizedLambdaExpressionSyntax node)
        {
            foreach (var param in node.ParameterList.Parameters) Visit(param);
            _returningFromCx.Push(new HashSet<int>());
            Visit(node.Body);
            _currentPathParents.UnionWith(_returningFromCx.Pop());
        }

        public override void VisitSimpleLambdaExpression(SimpleLambdaExpressionSyntax node)
        {
            Visit(node.Parameter);
            _returningFromCx.Push(new HashSet<int>());
            Visit(node.Body);
            _currentPathParents.UnionWith(_returningFromCx.Pop());
        }
        
                
        public override void VisitAccessorDeclaration(AccessorDeclarationSyntax node)
        {
            _returningFromCx.Push(new HashSet<int>());
            Visit(node.ExpressionBody);
            Visit(node.Body);
            _currentPathParents.UnionWith(_returningFromCx.Pop());
        }

        public override void VisitAssignmentExpression(AssignmentExpressionSyntax node)
        {
            Visit(node.Right);           
            Visit(node.Left);
        }

        public override void VisitInvocationExpression(InvocationExpressionSyntax node)
        {
            Visit(node.Expression);
            foreach (var param in node.ArgumentList.Arguments)
            {
                Visit(param);
            }      
        }

        public override void VisitUsingStatement(UsingStatementSyntax node)
        {
            Visit(node.Expression);
            Visit(node.Declaration);
            Visit(node.Statement);
        }

        public override void VisitYieldStatement(YieldStatementSyntax node)
        {
            // Treat like return statements but do not clear stack
            if (node.Expression != null)
            {
                Visit(node.Expression);
            }
            _returningFromCx.Peek().UnionWith(_currentPathParents);
        }

        public override void VisitReturnStatement(ReturnStatementSyntax node)
        {
            if (node.Expression != null)
            {
                Visit(node.Expression);                
            }
            _returningFromCx.Peek().UnionWith(_currentPathParents);
            _currentPathParents = new HashSet<int>();
        }

        public override void VisitElementAccessExpression(ElementAccessExpressionSyntax node)
        {
            foreach (var arg in node.ArgumentList.Arguments)
            {
                Visit(arg);
            }
            Visit(node.Expression);
        }

        public override void VisitConditionalExpression(ConditionalExpressionSyntax node)
        {
            Visit(node.Condition);
            RecordNodesInParallel(new List<SyntaxNode> { node.WhenTrue, node.WhenFalse });
        }

        public override void VisitVariableDeclarator(VariableDeclaratorSyntax node)
        {
            Visit(node.Initializer);
            VisitToken(node.Identifier);
            if (node.ArgumentList == null) return;

            foreach (var arg in node.ArgumentList.Arguments)
            {
                Visit(arg);
            }
            // We don't visit the types since we don't need this.
        }

        public override void VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            Visit(node.Initializer);
            VisitToken(node.Identifier);
            if (node.AccessorList == null) return;
            foreach (var accessor in node.AccessorList.Accessors)
            {
                Visit(accessor);
            }
        }
        
    }
}
