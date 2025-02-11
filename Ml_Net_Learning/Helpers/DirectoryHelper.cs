namespace Ml_Net_Learning.Helpers;

public class DirectoryHelper
{
    public static string GetFilePath(string fileName, string folderName)
    {
        string exeDirectory = AppDomain.CurrentDomain.BaseDirectory;
        var projectRoot = Path.GetFullPath(Path.Combine(exeDirectory, "..", "..", ".."));
        return  Path.Combine(projectRoot, folderName, fileName);
    }
}