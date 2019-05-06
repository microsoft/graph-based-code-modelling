using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;

namespace CSharpDatasetExtractionUtils
{
    /// <summary>
    /// A list that persists elements.
    /// </summary>
    public class PersistentResumeList
    {
        private readonly string _savePath;
        private readonly HashSet<string> _elements = new HashSet<string>();

        public PersistentResumeList(string savePath)
        {
            _savePath = savePath;
            if (File.Exists(savePath))
            {
                // Load
                foreach(var element in File.ReadAllLines(savePath).Where(l=>!string.IsNullOrEmpty(l)))
                {
                    _elements.Add(element);
                }
            }
            else
            {
                // Create
                File.WriteAllText(savePath, "");
            }
        }

        /// <summary>
        /// Add element to persistent list.
        /// </summary>
        /// <returns>False iff the element is already present, and true if it's a new element.</returns>
        [MethodImpl(MethodImplOptions.Synchronized)]
        public bool AddIfNotContained(string element)
        {
            if (!_elements.Add(element)) return false;
            File.AppendAllLines(_savePath, new[] { element });
            return true;
        }

        /// <summary>
        /// Add many elements at once to persistent list.
        /// </summary>
        public void BulkAdd(IEnumerable<string> newElements)
        {
            int num = 0;
            using (var saveFile = File.AppendText(_savePath))
            {
                foreach (var element in newElements)
                {
                    ++num;
                    _elements.Add(element);
                    saveFile.WriteLine(element);
                }
            }
        }
    }
}
