using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SourceGraphExtractionUtils
{
    public class ThreadSafeTextLog
    {
        private readonly object _lock = new object();
        private readonly string _logLocation;
        private readonly bool _alsoWriteToStdout;
        private readonly bool _logTime;

        public ThreadSafeTextLog(string logLocation, bool alsoWriteToStdout = true,
            bool append=false, bool logTime = true)
        {
            _logLocation = logLocation;
            _logTime = logTime;
            _alsoWriteToStdout = alsoWriteToStdout;
            if (!append && File.Exists(_logLocation))
            {
                File.Delete(_logLocation);
            }
        }

        public void LogMessage(string message)
            => LogMessage(new[] { message });

        public void LogMessage(IEnumerable<string> message)
        {
            lock (_lock)
            {
                if (_logTime)
                {
                    message = message.Select(m => $"[{DateTime.Now.ToString()}] {m}");
                }
                File.AppendAllLines(_logLocation, message);
            }
            if (_alsoWriteToStdout)
            {
                foreach (var msg in message)
                {
                    Console.WriteLine(msg);
                }
            }
        }
    }
}
