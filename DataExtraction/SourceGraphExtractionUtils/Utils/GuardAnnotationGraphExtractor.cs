using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System.Collections.Generic;
using System.Linq;

namespace SourceGraphExtractionUtils.Utils
{
    public class BlockGuardInformation
    {
        /// <summary>
        /// Guard information for enclosing block, potentially null if there is none.
        /// </summary>
        public readonly BlockGuardInformation EnclosingBlockInformation;

        private readonly SourceGraph _graph;

        private readonly List<(SyntaxNode guardNode, ISet<string> usedVariables)> _validatedGuards
            = new List<(SyntaxNode guardNode, ISet<string> usedVariables)>();

        private readonly List<(SyntaxNode guardNode, ISet<string> usedVariables)> _invalidatedGuards
            = new List<(SyntaxNode guardNode, ISet<string> usedVariables)>();

        /// <summary>
        /// SyntaxNodes of guards that were checked to hold for this block.
        /// </summary>
        public IEnumerable<(SyntaxNode guardNode, ISet<string> usedVariables)> ValidatedGuards => _validatedGuards;

        /// <summary>
        /// SyntaxNodes of guards that were checked to not hold in this block.
        /// </summary>
        public IEnumerable<(SyntaxNode guardNode, ISet<string> usedVariables)> InvalidatedGuards => _invalidatedGuards;

        public BlockGuardInformation(SourceGraph graph, BlockGuardInformation enclosingBlockInfo = null, SyntaxNode validatedGuard = null, SyntaxNode invalidatedGuard = null)
        {
            _graph = graph;
            EnclosingBlockInformation = enclosingBlockInfo;

            if (validatedGuard != null)
            {
                RecordValidatedGuard(validatedGuard);
            }

            if (invalidatedGuard != null)
            {
                RecordInvalidatedGuard(invalidatedGuard);
            }
        }

        private ISet<string> GetUsedVariables(SyntaxNode node)
            => new HashSet<string>(node.DescendantTokens().Where(tok => _graph.UsedVariableNodes.Contains(tok)).Select(tok => tok.Text));

        public void RecordValidatedGuard(SyntaxNode guardNode)
            => _validatedGuards.Add((guardNode: guardNode, usedVariables: GetUsedVariables(guardNode)));

        public void RecordValidatedGuard((SyntaxNode guardNode, ISet<string> usedVariables) validatedGuardInfo)
            => _validatedGuards.Add(validatedGuardInfo);

        public void RecordInvalidatedGuard(SyntaxNode guardNode)
            => _invalidatedGuards.Add((guardNode: guardNode, usedVariables: GetUsedVariables(guardNode)));

        public void RecordInvalidatedGuard((SyntaxNode guardNode, ISet<string> usedVariables) invalidatedGuardInfo)
            => _invalidatedGuards.Add(invalidatedGuardInfo);
    }

    public class GuardAnnotationGraphExtractor : CSharpSyntaxWalker
    {
        private readonly SourceGraph _graph;

        private BlockGuardInformation _blockGuardInformation;

        private GuardAnnotationGraphExtractor(SourceGraph graph)
        {
            _graph = graph;
        }

        public static void AddGuardAnnotationsToGraph(SourceGraph graph, SyntaxNode rootNode)
        {
            new GuardAnnotationGraphExtractor(graph).Visit(rootNode);
        }

        private void VisitVariable(string variableName, SyntaxToken identifierToken)
        {
            var curGuardInformation = _blockGuardInformation;
            while (curGuardInformation != null)
            {
                foreach (var validatedGuard in curGuardInformation.ValidatedGuards)
                {
                    _graph.AddEdge(identifierToken, SourceGraphEdge.GuardedBy, validatedGuard.guardNode);
                }

                foreach (var invalidatedGuard in curGuardInformation.InvalidatedGuards)
                {
                    _graph.AddEdge(identifierToken, SourceGraphEdge.GuardedByNegation, invalidatedGuard.guardNode);
                }

                curGuardInformation = curGuardInformation.EnclosingBlockInformation;
            }
        }

        public override void VisitIdentifierName(IdentifierNameSyntax node)
        {
            if (_graph.UsedVariableNodes.Contains(node.Identifier))
            {
                var variableName = node.ToString();
                VisitVariable(variableName, node.Identifier);
            }
        }

        #region Method (and related) declarations
        private void HandleBaseMethodDeclaration(BaseMethodDeclarationSyntax node)
        {
            // Roughly, set up a new block context, descent and then pop that context again.
            _blockGuardInformation = new BlockGuardInformation(_graph);
            Visit(node.Body);
            _blockGuardInformation = null;
        }

        public override void VisitConversionOperatorDeclaration(ConversionOperatorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override void VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override void VisitOperatorDeclaration(OperatorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);

        public override void VisitDestructorDeclaration(DestructorDeclarationSyntax node)
            => HandleBaseMethodDeclaration(node);
        #endregion

        #region Statements that create new conditional blocks (if, loops, ...)
        private void HandleConditionalBlock(IEnumerable<SyntaxNode> bodyNodes, SyntaxNode validatedGuard = null, SyntaxNode invalidatedGuard = null)
        {
            // Roughly, set up a new block context with condition, descent and then pop that context again.
            _blockGuardInformation = new BlockGuardInformation(_graph, enclosingBlockInfo: _blockGuardInformation,
                validatedGuard: validatedGuard, invalidatedGuard: invalidatedGuard);
            foreach (var bodyNode in bodyNodes)
            {
                Visit(bodyNode);
            }
            _blockGuardInformation = _blockGuardInformation.EnclosingBlockInformation;
        }

        public override void VisitIfStatement(IfStatementSyntax node)
        {
            Visit(node.Condition);
            HandleConditionalBlock(new[] { node.Statement }, validatedGuard: node.Condition);
            HandleConditionalBlock(new[] { node.Else }, invalidatedGuard: node.Condition);
        }

        public override void VisitSwitchStatement(SwitchStatementSyntax node)
        {
            foreach (var switchSection in node.Sections)
            {
                //TODO: Can we do something with the labels?
                HandleConditionalBlock(switchSection.Statements);
            }
        }

        public override void VisitForStatement(ForStatementSyntax node)
        {
            //Declaration and condition are unguarded, but rest is guarded:
            Visit(node.Declaration);
            Visit(node.Condition);
            HandleConditionalBlock(new SyntaxNode[] { node.Statement }.Concat(node.Incrementors), validatedGuard: node.Condition);
        }

        public override void VisitForEachStatement(ForEachStatementSyntax node)
        {
            Visit(node.Expression);
            if (_graph.UsedVariableNodes.Contains(node.Identifier))
            {
                // This is a special case required because ForEachStatements require a raw identifier
                // token (but matches what HandleConditionalBlock does):
                _blockGuardInformation = new BlockGuardInformation(_graph, enclosingBlockInfo: _blockGuardInformation,
                    validatedGuard: node.Expression, invalidatedGuard: null);
                var variableName = node.ToString();
                VisitVariable(variableName, node.Identifier);
                _blockGuardInformation = _blockGuardInformation.EnclosingBlockInformation;
            }
            HandleConditionalBlock(new SyntaxNode[] { node.Statement }, validatedGuard: node.Expression);
        }

        public override void VisitForEachVariableStatement(ForEachVariableStatementSyntax node)
        {
            Visit(node.Expression);
            HandleConditionalBlock(new SyntaxNode[] { node.Variable, node.Statement }, validatedGuard: node.Expression);
        }

        public override void VisitWhileStatement(WhileStatementSyntax node)
        {
            Visit(node.Condition);
            HandleConditionalBlock(new SyntaxNode[] { node.Statement }, validatedGuard: node.Condition);
        }

        public override void VisitDoStatement(DoStatementSyntax node)
        {
            //TODO: Can we do something with the condition, even if it's not checked in the first iteration?
            HandleConditionalBlock(new SyntaxNode[] { node.Statement });
            Visit(node.Condition);
        }
        #endregion

        #region Statements that modify control flow in conditional blocks (break, continue, return)
        private void HandleControlFlowBreak()
        {
            /* We know that if execution continues in this method after this return,
             * that's because the condition for this return did not hold. Thus, we can
             * consider the valid guards of the current lock as invalidated for its
             * enclosing block in the subsequent analysis (and similarly for invalidated
             * guards).
             */
            var enclosingBlockInformation = _blockGuardInformation?.EnclosingBlockInformation;
            if (enclosingBlockInformation != null)
            {
                foreach (var validatedGuard in _blockGuardInformation.ValidatedGuards)
                {
                    enclosingBlockInformation.RecordInvalidatedGuard(validatedGuard);
                }
                foreach (var invalidatedGuard in _blockGuardInformation.InvalidatedGuards)
                {
                    enclosingBlockInformation.RecordValidatedGuard(invalidatedGuard);
                }
            }
        }

        public override void VisitReturnStatement(ReturnStatementSyntax node)
        {
            Visit(node.Expression);
            HandleControlFlowBreak();
        }

        public override void VisitBreakStatement(BreakStatementSyntax node) => HandleControlFlowBreak();
        public override void VisitContinueStatement(ContinueStatementSyntax node) => HandleControlFlowBreak();
        #endregion
    }
}
