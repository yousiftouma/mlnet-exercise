using System.IO;

namespace OnnxObjectDetectionWeb.Utilities
{
    public static class CommonHelpers
    {
        /// <summary>
        /// Gets an absolute path for the given relative path.
        /// </summary>
        /// <param name="relativePath">
        /// Relative path to generate absolute path for.
        /// Root is where the executed assemblies lie (typically the build folder bin/Debug/netcoreapp3.1 when running locally)
        /// </param>
        public static string GetAbsolutePath(string relativePath)
        {            
            var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            var assemblyFolderPath = dataRoot.Directory.FullName;

            var fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
