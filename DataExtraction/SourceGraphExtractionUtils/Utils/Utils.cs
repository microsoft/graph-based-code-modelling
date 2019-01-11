using System.Collections.Generic;

namespace SourceGraphExtractionUtils.Utils
{
    public class Utils
    {
        private static System.Random _rng = new System.Random();

        private static void ListElementSwap<T>(List<T> data, int i, int j)
        {
            T item = data[i];
            data[i] = data[j];
            data[j] = item;
        }

        public static void KnuthShuffle<T>(List<T> data)
        {
            for (var i = 0; i < data.Count - 2; ++i)
            {
                ListElementSwap(data, i, _rng.Next(i, data.Count));
            }
        }
    }
}
