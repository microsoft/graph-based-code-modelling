using System;
using System.Collections.Generic;
using System.IO;

namespace CSharpDatasetExtractionUtils
{
    public class Utils
    {
        private static Random _rng = new Random();

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
 
        public static string GetRelativePath(string fullPath, string rootPath)
        {
            var fullPathUri = new Uri(fullPath);

            // Folders must end in a slash
            if (!rootPath.EndsWith(Path.DirectorySeparatorChar.ToString()))
            {
                rootPath += Path.DirectorySeparatorChar;
            }

            var rootPathUri = new Uri(Path.GetFullPath(rootPath));
            return rootPathUri.MakeRelativeUri(fullPathUri).ToString().Replace('/', Path.DirectorySeparatorChar);
        }
    }
}
