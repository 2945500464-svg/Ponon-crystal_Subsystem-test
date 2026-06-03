Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = appDir
command = "cmd /c pythonw.exe """ & appDir & "\main.py"" || python.exe """ & appDir & "\main.py"""
shell.Run command, 0, False
