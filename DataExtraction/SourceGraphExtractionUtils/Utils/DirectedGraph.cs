using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace SourceGraphExtractionUtils.Utils
{
    public struct Edge<TNode, TEdgeLabel>
    {
        public readonly TNode Source;
        public readonly TNode Target;
        public readonly TEdgeLabel Label;

        public Edge(TNode source, TEdgeLabel label, TNode target)
        {
            Source = source;
            Label = label;
            Target = target;
        }

        public override string ToString()
            => $"{Source} --{Label}--> {Target}";

        public void Deconstruct(out TNode sourceNode, out TEdgeLabel label, out TNode targetNode)
        {
            sourceNode = Source;
            label = Label;
            targetNode = Target;
        }
    }

    public class DirectedGraph<TNode, TEdgeLabel>
    {
        private readonly IEqualityComparer<TNode> _nodeComparer;
        private readonly IEqualityComparer<Edge<TNode, TEdgeLabel>> _edgeComparer;
        /// <summary>
        /// Maps node to a (inEdge, outEdges) pair for that node.
        /// </summary>
        private readonly Dictionary<TNode, (HashSet<Edge<TNode, TEdgeLabel>> inEdges, HashSet<Edge<TNode, TEdgeLabel>> outEdges)> _edges;

        public DirectedGraph(IEqualityComparer<TNode> nodeComparer = null, IEqualityComparer<Edge<TNode, TEdgeLabel>> edgeComparer = null) {
            _nodeComparer = nodeComparer ?? EqualityComparer<TNode>.Default;
            _edgeComparer = edgeComparer ?? EqualityComparer<Edge<TNode, TEdgeLabel>>.Default;
            _edges = new Dictionary<TNode, (HashSet<Edge<TNode, TEdgeLabel>> inEdges, HashSet<Edge<TNode, TEdgeLabel>> outEdges)> (_nodeComparer);
        }

        public IEnumerable<TNode> Nodes => _edges.Keys;

        public int CountInEdges => _edges.Sum(kv => kv.Value.inEdges.Count);

        public int CountOutEdges => _edges.Sum(kv => kv.Value.outEdges.Count);

        public int CountNodes => _edges.Count;

        public bool AddEdge(Edge<TNode, TEdgeLabel> edge)
        {
            AddNode(edge.Source);
            AddNode(edge.Target);
            bool outWasAdded = true;
            if (_edges.TryGetValue(edge.Source, out var sourceEdges))
            {
                outWasAdded = sourceEdges.outEdges.Add(edge);
            }
            else
            {
                sourceEdges = (inEdges: new HashSet<Edge<TNode, TEdgeLabel>>(_edgeComparer),
                               outEdges: new HashSet<Edge<TNode, TEdgeLabel>>(_edgeComparer) { edge });
                _edges.Add(edge.Source, sourceEdges);
            }

            bool inWasAdded = true;
            if (_edges.TryGetValue(edge.Target, out var targetEdges))
            {
                inWasAdded = targetEdges.inEdges.Add(edge);
            }
            else
            {
                targetEdges = (inEdges: new HashSet<Edge<TNode, TEdgeLabel>>(_edgeComparer) { edge },
                               outEdges: new HashSet<Edge<TNode, TEdgeLabel>>(_edgeComparer));
                _edges.Add(edge.Target, targetEdges);
            }

            Debug.Assert(inWasAdded == outWasAdded);
            return inWasAdded;
        }

        public bool AddEdge(TNode source, TEdgeLabel label, TNode target)
            => AddEdge(new Edge<TNode, TEdgeLabel>(source, label, target));

        public bool AddEdges<TNodeSubtype>(TNode source, TEdgeLabel label, IEnumerable<TNodeSubtype> targets)
            where TNodeSubtype : TNode
            => targets.Aggregate(false, (addedEdge, targetNode) => AddEdge(source, label, targetNode) || addedEdge);

        public bool AddEdges<TNodeSubtype>(IEnumerable<TNodeSubtype> sources, TEdgeLabel label, TNode target)
            where TNodeSubtype : TNode
            => sources.Aggregate(false, (addedEdge, sourceNode) => AddEdge(sourceNode, label, target) || addedEdge);

        public bool AddNode(TNode node)
        {
            if (!_edges.ContainsKey(node))
            {
                if (_edges.Count > ExtractionLimits.MAX_NUM_ALLOWED_NODES)
                {
                    throw new Exception("Exceeded maximum number of nodes that can be handled!");
                }

                //Console.WriteLine($"Adding node of type {node.GetType().Name}: {node}");
                _edges.Add(node, (inEdges: new HashSet<Edge<TNode, TEdgeLabel>>(_edgeComparer),
                                  outEdges: new HashSet<Edge<TNode, TEdgeLabel>>(_edgeComparer)));
                return true;
            }
            return false;
        }

        public bool ContainsNode(TNode node)
        {
            return _edges.ContainsKey(node);
        }

        public void RemoveNode(TNode node)
        {
            //Console.WriteLine($"Removing node of type {node.GetType().Name}: {node}");
            (var nodeInEdges, var nodeOutEdges) = GetEdges(node);

            //Remove all references to the node in incoming/outgoing edge sets:
            foreach (var edge in nodeInEdges)
            {
                (var sourceInEdges, var sourceOutEdges) = _edges[edge.Source];
                _edges[edge.Source] = (sourceInEdges,
                                       new HashSet<Edge<TNode, TEdgeLabel>>(sourceOutEdges.Where(e => !_nodeComparer.Equals(e.Target, node)), _edgeComparer));
                
            }

            foreach (var edge in nodeOutEdges)
            {
                (var targetInEdges, var targetOutEdges) = _edges[edge.Target];
                _edges[edge.Target] = (new HashSet<Edge<TNode, TEdgeLabel>>(targetInEdges.Where(e => !_nodeComparer.Equals(e.Source, node)), _edgeComparer),
                                       targetOutEdges);
            }

            _edges.Remove(node);
        }

        public void RemoveEdge(Edge<TNode, TEdgeLabel> edge)
        {
            if (_edges.TryGetValue(edge.Source, out var sourceEdges))
            {
                sourceEdges.outEdges.Remove(edge);
            }
            if (_edges.TryGetValue(edge.Target, out var targetEdges))
            {
                targetEdges.inEdges.Remove(edge);
            }
        }

        public (IEnumerable<Edge<TNode, TEdgeLabel>> inEdges, IEnumerable<Edge<TNode, TEdgeLabel>> outEdges) GetEdges(TNode node)
            => _edges.TryGetValue(node, out var nodeEdges) ? nodeEdges 
                                                           : (Enumerable.Empty<Edge<TNode, TEdgeLabel>>(), Enumerable.Empty<Edge<TNode, TEdgeLabel>>());

        public IEnumerable<Edge<TNode, TEdgeLabel>> GetInEdges(TNode node)
            => _edges.TryGetValue(node, out var nodeEdges) ? nodeEdges.inEdges : Enumerable.Empty<Edge<TNode, TEdgeLabel>>();

        public IEnumerable<Edge<TNode, TEdgeLabel>> GetOutEdges(TNode node)
            => _edges.TryGetValue(node, out var nodeEdges) ? nodeEdges.outEdges : Enumerable.Empty<Edge<TNode, TEdgeLabel>>();

        #region Graph outputs
        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach ((var node, var (_, outEdges)) in _edges)
            {
                sb.Append("Edges from ")
                  .Append(node)
                  .Append(":");
                bool afterFirst = false;
                foreach (var edge in outEdges)
                {
                    if (afterFirst)
                    {
                        sb.Append(", ");
                    }
                    sb.Append("--")
                      .Append(edge.Label)
                      .Append("-->")
                      .Append(edge.Target.ToString());
                    afterFirst = true;
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }

        private static void WriteNodeJson(JsonWriter jWriter, Dictionary<TNode, int> nodeNumberer, TNode node)
        {
            if (!nodeNumberer.TryGetValue(node, out var nodeNumber))
            {
                nodeNumber = nodeNumberer.Count;
                nodeNumberer[node] = nodeNumber;
            }
            jWriter.WriteValue(nodeNumber);
        }

        public void WriteJson(JsonWriter jWriter, Dictionary<TNode, int> nodeNumberer,
            IEnumerable<Action<JsonWriter, Dictionary<TNode, int>>> additionalNodeInfoWriters,
            IEnumerable<(Predicate<TEdgeLabel> acceptsEdgeLabel, Action<JsonWriter, (TNode From, TNode To)> writer)> additionalEdgeInfoWriters)
        {
            jWriter.WriteStartObject();

            // Group by edge type
            var edgesByType = new Dictionary<TEdgeLabel, List<(TNode From, TNode To)>>();
            foreach (var edge in Nodes.SelectMany(n => GetOutEdges(n)))
            {
                if (!edgesByType.TryGetValue(edge.Label, out var edgeList))
                {
                    edgeList = new List<(TNode from, TNode to)>();
                    edgesByType.Add(edge.Label, edgeList);
                }
                edgeList.Add((edge.Source, edge.Target));
            }

            // Write Edges
            jWriter.WritePropertyName("Edges");
            
            jWriter.WriteStartObject();
            foreach (var (edgeType, edges) in edgesByType)
            {
                jWriter.WritePropertyName(edgeType.ToString());
                jWriter.WriteStartArray();
                foreach (var (From, To) in edges)
                {
                    jWriter.WriteStartArray();
                    WriteNodeJson(jWriter, nodeNumberer, From);
                    WriteNodeJson(jWriter, nodeNumberer, To);
                    jWriter.WriteEndArray();
                }
                jWriter.WriteEndArray();
            }
            jWriter.WriteEndObject();

            // Write Edge Values, if any
            jWriter.WritePropertyName("EdgeValues");
            jWriter.WriteStartObject();
            foreach (var (edgeType, edges) in edgesByType)
            {
                var usedEdgeInfoWriters = additionalEdgeInfoWriters
                    .Where(edgeWriter => edgeWriter.acceptsEdgeLabel(edgeType))
                    .Select(e => e.writer).ToArray();
                if (usedEdgeInfoWriters.Length == 0)
                {
                    continue;
                }

                jWriter.WritePropertyName(edgeType.ToString());
                jWriter.WriteStartArray();
                foreach (var (From, To) in edges)
                {
                    jWriter.WriteStartArray();
                    foreach (var edgeInfoWriter in usedEdgeInfoWriters)
                    {
                        edgeInfoWriter(jWriter, (From, To));
                    }
                    jWriter.WriteEndArray();
                }
                jWriter.WriteEndArray();
            }
            jWriter.WriteEndObject();

            foreach (var additionalWriter in additionalNodeInfoWriters)
            {
                additionalWriter(jWriter, nodeNumberer);
            }

            jWriter.WriteEndObject();
        }

        public void ToDotFile(string outputPath, Dictionary<TNode, object> nodeLabeler, Func<TNode, string> nodeShaper, Func<TNode, string> nodeRenderer, bool diffable=false)
        {
            using (var fileStream = File.Create(outputPath))
            using (var textStream = new StreamWriter(fileStream))
            {
                textStream.WriteLine("digraph program {");
                foreach (var node in Nodes)
                {
                    textStream.WriteLine("  node{0}[shape={1} label=\"{0}: {2}\"];", nodeLabeler[node], nodeShaper(node), nodeRenderer(node));

                    foreach (var edge in GetOutEdges(node))
                    {
                        textStream.WriteLine("  node{0} -> node{1} [label=\"{2}\"];", nodeLabeler[edge.Source], nodeLabeler[edge.Target], edge.Label.ToString());
                    }
                }
                textStream.WriteLine("}");
            }
        }

        public Dictionary<TNode, int> GetNodeNumberer()
        {
            Dictionary<TNode, int> nodeNumberer = new Dictionary<TNode, int>();
            foreach (var node in Nodes)
            {
                if (!nodeNumberer.TryGetValue(node, out var nodeNumber))
                {
                    nodeNumber = nodeNumberer.Count;
                    nodeNumberer[node] = nodeNumber;
                }
            }
            return nodeNumberer;
        }
        #endregion
    }
}
 