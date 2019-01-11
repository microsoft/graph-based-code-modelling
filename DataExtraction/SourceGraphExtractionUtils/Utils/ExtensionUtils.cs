using System.Collections.Generic;

namespace SourceGraphExtractionUtils.Utils
{
    public static class ExtensionUtils
    {
        public static void Deconstruct<K, V>(this KeyValuePair<K, V> kvp, out K key, out V value)
        {
            key = kvp.Key;
            value = kvp.Value;
        }
    }
}
