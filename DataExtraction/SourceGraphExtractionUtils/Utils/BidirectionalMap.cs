using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SourceGraphExtractionUtils.Utils
{
    [Serializable()]
    public class BidirectionalMap<K, V>
        where V : class
    {
        private readonly Dictionary<K, V> forwardMap;
        private readonly Dictionary<V, K> backwardsMap;

        public int Count => forwardMap.Count;

        public BidirectionalMap()
        {
            forwardMap = new Dictionary<K, V>();
            backwardsMap = new Dictionary<V, K>();
        }

        public BidirectionalMap(IEqualityComparer<V> valueComparer = null)
        {
            forwardMap = new Dictionary<K, V>();

            if (valueComparer != null) backwardsMap = new Dictionary<V, K>(valueComparer);
            else backwardsMap = new Dictionary<V, K>();
        }

        public void Add(K key, V value)
        {
            forwardMap.Add(key, value);
            backwardsMap.Add(value, key);
        }

        public V GetValue(K key)
        {
            return forwardMap[key];
        }

        public K GetKey(V value)
        {
            return backwardsMap[value];
        }

        public bool Contains(K key)
        {
            return forwardMap.ContainsKey(key);
        }

        public bool TryGetValue(K key, out V value)
        {
            return forwardMap.TryGetValue(key, out value);
        }

        public bool TryGetKey(V value, out K key)
        {
            return backwardsMap.TryGetValue(value, out key);
        }

        public bool Contains(V value)
        {
            return backwardsMap.ContainsKey(value);
        }

        public void Delete(K key)
        {
            V value = forwardMap[key];
            forwardMap.Remove(key);
            backwardsMap.Remove(value);
        }

        public void Delete(V value)
        {
            K key = backwardsMap[value];
            forwardMap.Remove(key);
            backwardsMap.Remove(value);
        }
    }
}
