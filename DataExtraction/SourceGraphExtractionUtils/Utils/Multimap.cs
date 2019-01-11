using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SourceGraphExtractionUtils.Utils
{
    class Multimap<K, V>
    {
        private readonly Dictionary<K, HashSet<V>> _elements = new Dictionary<K, HashSet<V>>();

        public Multimap() {}

        public Multimap(IEnumerable<IGrouping<K, V>> input)
        {
            foreach (var grouping in input)
            {
                var key = grouping.Key;
                AddMany(key, grouping);
            }
        }

        public Multimap(Dictionary<K, ISet<V>> input)
        {
            foreach (var grouping in input)
            {
                var key = grouping.Key;
                AddMany(key, grouping.Value);
            }
        }

        public void Add(K key, V value)
        {
            HashSet<V> keyElements;
            if (!_elements.TryGetValue(key, out keyElements))
            {
                keyElements = new HashSet<V>();
                _elements.Add(key, keyElements);
            }
            keyElements.Add(value);
        }

        public Multimap<V, K> Invese()
        {
            var inverse = new Multimap<V, K>();
            foreach (var pair in this.KeyValuePairs())
            {
                inverse.Add(pair.Item2, pair.Item1);
            }
            return inverse;
        }

        public void AddMany(K key, IEnumerable<V> values)
        {
            HashSet<V> keyElements;
            if (!_elements.TryGetValue(key, out keyElements))
            {
                keyElements = new HashSet<V>();
                _elements.Add(key, keyElements);
            }
            keyElements.UnionWith(values);
        } 

        public IEnumerable<V> Values(K key)
        {
            HashSet<V> keyElements;
            if (!_elements.TryGetValue(key, out keyElements))
            {
                return Enumerable.Empty<V>();
            }
            return keyElements.AsEnumerable();
        }

        public bool ContainsEntry(K key, V value)
        {
            HashSet<V> keyElements;
            if (!_elements.TryGetValue(key, out keyElements))
            {
                return false;
            }
            return keyElements.Contains(value);
        }

        public IEnumerable<K> Keys(K key)
        {
            return _elements.Keys;
        }

        public IEnumerable<Tuple<K, V>> KeyValuePairs()
        {
            foreach(var keyset in _elements)
            {
                foreach(var value in keyset.Value)
                {
                    yield return Tuple.Create(keyset.Key, value);
                }
            }
        }

        public int CountFor(K key)
        {
            HashSet<V> keyElements;
            if (!_elements.TryGetValue(key, out keyElements))
            {
                return 0;
            }
            return keyElements.Count;
        }


    }
}
