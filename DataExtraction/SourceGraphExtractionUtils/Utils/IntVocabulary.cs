using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SourceGraphExtractionUtils.Utils
{
    class IntVocabulary<T> where T : class
    {
        private readonly BidirectionalMap<int, T> _dictionary = new BidirectionalMap<int, T>();
        private int _nextId = 0;

        public int Count => _dictionary.Count;

        public int Get(T obj, bool addIfNotPresent=false)
        {
            int key;
            if (!_dictionary.TryGetKey(obj, out key))
            {
                if (!addIfNotPresent)
                {
                    throw new Exception("Object not in vocabulary");
                }
                key = _nextId;
                _dictionary.Add(key, obj);
                _nextId++;
            }
            return key;
        }

        public bool Contains(T obj)
        {
            return _dictionary.Contains(obj);
        }

        public T Get(int objId)
        {
            return _dictionary.GetValue(objId);
        }
    }
}
