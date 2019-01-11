using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SourceGraphExtractionUtils.Utils
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
        
        public bool AddIfNotContained(string element)
        {
            lock (this)
            {
                if (!_elements.Add(element)) return false;
                File.AppendAllLines(_savePath, new[] { element });
                return true;
            }
        }
    }
}
