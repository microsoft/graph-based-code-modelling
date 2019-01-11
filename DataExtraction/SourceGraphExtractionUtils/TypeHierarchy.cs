using SourceGraphExtractionUtils.Utils;
using Microsoft.CodeAnalysis;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Diagnostics;

namespace SourceGraphExtractionUtils
{
    /// <summary>
    /// A class that encapsulates all type hierarchy.
    /// </summary>
    public class TypeHierarchy
    {
        private readonly IntVocabulary<string> _typeDictionary = new IntVocabulary<string>();
        private readonly Multimap<int, int> _typeParents = new Multimap<int, int>();

        private readonly object _lock = new object();

        public void Add(ITypeSymbol type)
        {
            lock (_lock)
            {
                if (type.IsReferenceType)
                {
                    // This is definitely an object
                    AddAsObject(type);
                }

                if (type is IArrayTypeSymbol)
                {
                    var arrayTypeSymbol = type as IArrayTypeSymbol;
                    Add(arrayTypeSymbol.ElementType);
                }

                if (type is ITypeParameterSymbol)
                {
                    var typeParam = type as ITypeParameterSymbol;
                    if (typeParam.Variance == VarianceKind.In)
                    {
                        // this generic will accept all supertypes, not sure how to use.
                    }
                    else if (typeParam.Variance == VarianceKind.Out)
                    {
                        // Probably nothing should happen here.
                    }

                    if (typeParam.HasReferenceTypeConstraint)
                    {
                        AddAsObject(type);
                    }
                    else
                    {
                        // Add as a special "TypeParam" top type
                        AddAsTypeParam(type);
                    }
                }

                if (type.BaseType != null)
                {
                    var addBaseType = Add(type, type.BaseType);
                    if (addBaseType) Add(type.BaseType);
                }

                foreach (var implementedIface in type.Interfaces)
                {
                    var addIface = Add(type, implementedIface);
                    if (addIface) Add(implementedIface);
                }

                if (type is INamedTypeSymbol)
                {
                    var namedType = type as INamedTypeSymbol;
                    if (namedType.IsGenericType)
                    {
                        if (namedType.ConstructedFrom != type)
                        {
                            var addErasure = Add(type, namedType.ConstructedFrom);
                            if (addErasure) Add(namedType.ConstructedFrom);
                        }
                    }
                }
            }
        }

        private bool Add(ITypeSymbol subtype, ITypeSymbol type)
        {
            var baseTypeExisted = _typeDictionary.Contains(type.ToString());
            int subtypeId = _typeDictionary.Get(subtype.ToString(), addIfNotPresent: true);
            int typeId = _typeDictionary.Get(type.ToString(), addIfNotPresent: true);

            if (subtype.ToString() == type.ToString()) return baseTypeExisted;

            _typeParents.Add(subtypeId, typeId);
            return !baseTypeExisted;
        }

        private void AddAsObject(ITypeSymbol subtype)
        {
            int subtypeId = _typeDictionary.Get(subtype.ToString(), addIfNotPresent: true);
            int typeId = _typeDictionary.Get("object", addIfNotPresent: true);
            _typeParents.Add(subtypeId, typeId);
        }

        private void AddAsTypeParam(ITypeSymbol subtype)
        {
            int subtypeId = _typeDictionary.Get(subtype.ToString(), addIfNotPresent: true);
            int typeId = _typeDictionary.Get("<typeParam>", addIfNotPresent: true);
            _typeParents.Add(subtypeId, typeId);
        }
        
        public void SaveTypeHierarchy(string outputFilename)
        {
            lock (_lock)
            {
                using (var fileStream = File.Create(outputFilename))
                using (var gzipStream = new GZipStream(fileStream, CompressionMode.Compress, false))
                using (var textStream = new StreamWriter(gzipStream))
                {
                    var serializer = new JsonSerializer { NullValueHandling = NullValueHandling.Ignore };
                    serializer.Serialize(textStream, new SerializableHierarchy(this));
                }
            }
        }

        public static bool ComputeTypeForSymbol(ISymbol symbol, out ITypeSymbol res)
        {
            switch (symbol)
            {
                case IParameterSymbol paramSym:
                    res = paramSym.Type;
                    return true;
                case ILocalSymbol localSym:
                    res = localSym.Type;
                    return true;
                case IFieldSymbol fieldSym:
                    res = fieldSym.Type;
                    return true;
                case IEventSymbol eventSym:
                    res = eventSym.Type;
                    return true;
                case IPropertySymbol propSym:
                    res = propSym.Type;
                    return true;
                default:
                    res = null;
                    return false;
            }
        }

        public List<string> ComputeAndAddSymbolTypes(IEnumerable<ISymbol> variableSymbols)
        {
            lock (_lock)
            {
                var typeNames = new List<string>();
                foreach (var symbol in variableSymbols)
                {
                    if (ComputeTypeForSymbol(symbol, out var typeSymbol))
                    {
                        Add(typeSymbol);
                        typeNames.Add(typeSymbol.ToString());
                    }
                    else
                    {
                        throw new Exception("Symbol not of recognized type: " + symbol);
                    }
                }
                return typeNames;
            }
        }

        private class SerializableHierarchy
        {
            public List<string> types = new List<string>();
            public List<HashSet<int>> outgoingEdges = new List<HashSet<int>>();

            public SerializableHierarchy(TypeHierarchy typeHierarchy)
            {
                for (int i = 0; i < typeHierarchy._typeDictionary.Count; i++) {
                    types.Add(typeHierarchy._typeDictionary.Get(i));
                    outgoingEdges.Add(new HashSet<int>(typeHierarchy._typeParents.Values(i)));
               }
            }
        }

        public static TypeHierarchy Load(string filename)
        {
            using (var fileStream = File.OpenRead(filename))
            using (var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress))
            using (var textStream = new StreamReader(gzipStream))
            {
                var serializer = new JsonSerializer { NullValueHandling = NullValueHandling.Ignore };
                var deserializedHier = (SerializableHierarchy)serializer.Deserialize(textStream, typeof(SerializableHierarchy));

                var typeHierarchy = new TypeHierarchy();
                for(int i=0; i < deserializedHier.types.Count; i++)
                {
                    var rId = typeHierarchy._typeDictionary.Get(deserializedHier.types[i], addIfNotPresent: true);
                    Debug.Assert(rId == i);
                    typeHierarchy._typeParents.AddMany(i, deserializedHier.outgoingEdges[i]);
                }
                return typeHierarchy;
            }
        }
    }
}
