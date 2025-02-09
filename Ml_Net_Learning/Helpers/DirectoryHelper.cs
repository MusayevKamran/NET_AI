namespace Ml_Net_Learning.Helpers;

public class DirectoryHelper
{
    public static string GetProjectRoot(string fileName)
    {
        string exeDirectory = AppDomain.CurrentDomain.BaseDirectory;
        var projectRoot = Path.GetFullPath(Path.Combine(exeDirectory, "..", "..", ".."));
        return  Path.Combine(projectRoot, "CSV", fileName);
    }
}