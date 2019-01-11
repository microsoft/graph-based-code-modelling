using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestProject
{
    class Program
    {
        static int classVar = 23;

        static void Main(string[] args) {
            var foo = args[0];
            var baz = - 42;
            var intVar = args.Length;

            intVar = intVar + classVar;
            
            if (foo.StartsWith("foobar") && foo.EndsWith(Environment.NewLine, StringComparison.Ordinal)) {
                var l = foo.Length + 10;
            }
            else
            {
                var bar = foo.IndexOf('-');
                bar = foo.IndexOf('-', 2, 43);
            }
        }

        int Foo(int i, bool b)
        {
            int[][] arr = new int[][] { null, new int [] { 1, 2, 3, 4 } };
            int[] iarr = arr[1];
            for (var j = i; j > classVar2; --j)
            {
                if (j > 0 && j < iarr.Length) {
                    i = iarr[j] * -1 + 4;
                }
                b = !b;
                if (j > 4) return j;
                if (4 < classVar2) { classVar2++; } else { return j; }
                classVar2 += j;
            }

            if (b)
            {
                return b ? 2 : i;
            }
            else
            {
                return b ? 1 : -i;
            }
        }
        
        int classVar2 = 20 + 3;

        void Bar() {
            string a = "a";
            string b = a;
            if (a == null) return;
            if (b == null) return;
            return;
        }
    }
}
