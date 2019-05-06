using DataExtraction.Utils;
using SourceGraphExtractionUtils.Utils;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace SourceGraphExtractionUtils
{
    public class PhogExtractor : IDisposable
    {
        private readonly ChunkedJsonGzWriter _writer;
        private readonly int _maxContextStepsForAncestorContext;
        private readonly string _repositoryRootPath;

        public PhogExtractor(string outputFilePath, int maxContextStepsForAncestorContext, string repositoryRootPath)
        {
            _writer = new ChunkedJsonGzWriter(outputFilePath, useJsonlFormat: true, resumeIfFilesExist: true);
            _maxContextStepsForAncestorContext = maxContextStepsForAncestorContext;
            _repositoryRootPath = repositoryRootPath;
        }

        public void ExtractFor(SyntaxNode targetNode)
        {
            SyntaxNode parentContext = targetNode.Ancestors().Take(_maxContextStepsForAncestorContext).Last();
            _writer.WriteElement(jw => WriteJsonFor(jw, targetNode, parentContext));
        }

        private void WriteJsonFor(JsonWriter jw, SyntaxNode targetNode, SyntaxNode parentContext)
        {
            jw.WriteStartObject();
            jw.WritePropertyName("Filename");
            if (targetNode.SyntaxTree.FilePath.StartsWith(_repositoryRootPath))
            {
                jw.WriteValue(targetNode.SyntaxTree.FilePath.Substring(_repositoryRootPath.Length));
            }
            else
            {
                jw.WriteValue(targetNode.SyntaxTree.FilePath);
            }
            jw.WritePropertyName("HoleSpan");
            jw.WriteValue(targetNode.FullSpan.ToString());

            jw.WritePropertyName("HoleLineSpan");
            jw.WriteValue(targetNode.SyntaxTree.GetLineSpan(targetNode.Span).Span.ToString());

            jw.WritePropertyName("OriginalExpression");
            jw.WriteValue(targetNode.ToString());

            jw.WritePropertyName("PhogData");
            int generationStartingPoint = WritePhogData(jw, parentContext, targetNode);

            jw.WritePropertyName("GenerationStartingId");
            jw.WriteValue(generationStartingPoint);

            jw.WriteEndObject();
        }

        private static bool DescendIntoChild(SyntaxNode node)
        {
            if (node.IsKind(SyntaxKind.IdentifierName))
            {
                return false;
            }
            return true;
        }

        private int WritePhogData(JsonWriter jw, SyntaxNode parentContext, SyntaxNode targetNode)
        {
            Dictionary<SyntaxNodeOrToken, int> nodeToIdx = new Dictionary<SyntaxNodeOrToken, int>();
            List<SyntaxNodeOrToken> preOrderNodeSequence = new List<SyntaxNodeOrToken>();
            int idx = 0;
            bool hasVisitedTarget = false;
            foreach (var nodeOrToken in parentContext.DescendantNodesAndTokensAndSelf(descendIntoChildren: DescendIntoChild))
            {
                if (nodeOrToken.IsNode)
                {
                    var node = nodeOrToken.AsNode();
                    if (hasVisitedTarget && !node.Ancestors().Contains(targetNode))
                    {
                        break;  // we just moved out of the expression, so we don't want to do anything else.
                    }
                    if (node == targetNode) hasVisitedTarget = true;
                }
                    
                nodeToIdx[nodeOrToken] = idx++;
                preOrderNodeSequence.Add(nodeOrToken);                      
            }

            jw.WriteStartArray();
            foreach (var node in preOrderNodeSequence)
            {
                NodeToJson(jw, node, nodeToIdx);
            }
            jw.WriteEndArray();

            return nodeToIdx[targetNode];
        }

        private void NodeToJson(JsonWriter jw, SyntaxNodeOrToken node, Dictionary<SyntaxNodeOrToken, int> nodeToIdx)
        {
            jw.WriteStartObject();

            jw.WritePropertyName("id");
            jw.WriteValue(nodeToIdx[node]);

            jw.WritePropertyName("type");
            jw.WriteValue(node.Kind().ToString());

            if (node.IsKind(SyntaxKind.IdentifierName) || node.IsKind(SyntaxKind.PredefinedType) || RoslynUtils.IsSimpleLiteral(node) || node.IsToken || node.AsNode().ChildNodes().Count() == 0)
            {
                jw.WritePropertyName("value");
                jw.WriteValue(node.ToString());
            }
            else
            {
                jw.WritePropertyName("children");
                jw.WriteStartArray();
                foreach(var child in node.AsNode().ChildNodesAndTokens())
                {
                    if (!nodeToIdx.TryGetValue(child, out int idx)) idx = int.MaxValue;
                    jw.WriteValue(idx);
                }
                jw.WriteEndArray();
            }

            jw.WriteEndObject();
        }

        public void Dispose()
        {
            _writer.Dispose();
        }
    }
}
