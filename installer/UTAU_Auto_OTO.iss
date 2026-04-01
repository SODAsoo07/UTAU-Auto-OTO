#define MyAppName "UTAU Auto OTO"
#define MyAppPublisher "UTAU Auto OTO"
#define MyAppExeName "UTAU_Auto_OTO\\UTAU_Auto_OTO.exe"

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#ifndef AppChannel
  #define AppChannel "stable"
#endif

#ifndef SourceDir
  #define SourceDir "..\\UTAU_Auto_OTO_Release_stable"
#endif

#ifndef OutputDir
  #define OutputDir "..\\installer_output"
#endif

#ifndef OutputNameSuffix
  #define OutputNameSuffix ""
#endif

#ifndef ExcludeMfaRuntimeBundle
  #define ExcludeMfaRuntimeBundle "0"
#endif

#define InternalTestScriptExcludes "scripts\\*.py,scripts\\*.ps1,scripts\\*.bat,scripts\\*.cmd,build_alignment_test_folder.py,compare_alignment_visual.py,export_textgrid_to_sinsy_lab.py,preprocess_oto_cv_for_sequence_training.py,preprocess_sinsy_labels_for_sequence_training.py,train_sequence_aligner_profile_from_sinsy.py,sandbox_smoke_check.ps1"

#if ExcludeMfaRuntimeBundle == "1"
  #define ReleaseFileExcludes "config.json,.mfa_startup_repair_state.json,.cuda_startup_check_state.json,logs\\*,mfa_runtime_bundle\\*,{#InternalTestScriptExcludes}"
#else
  #define ReleaseFileExcludes "config.json,.mfa_startup_repair_state.json,.cuda_startup_check_state.json,logs\\*,{#InternalTestScriptExcludes}"
#endif

#ifdef SetupIconFilePath
  #define SetupIconFileParam SetupIconFilePath
#else
  #ifexist "..\\release_assets\\AutoOTO-icon.ico"
    #define SetupIconFileParam "..\\release_assets\\AutoOTO-icon.ico"
  #endif
#endif

[Setup]
AppId={{A84501A4-413A-4ED8-89CD-BF5D816A4CE2}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\UTAU_Auto_OTO
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
PrivilegesRequired=lowest
Compression=lzma2
SolidCompression=yes
; MFA runtime/model bundled builds can exceed single-EXE Windows limits.
; Enable disk spanning so Inno emits Setup.exe + *.bin slices.
DiskSpanning=yes
DiskSliceSize=2000000000
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=UTAU_Auto_OTO_Setup_{#AppVersion}_{#AppChannel}{#OutputNameSuffix}
UninstallDisplayIcon={app}\{#MyAppExeName}
DisableProgramGroupPage=yes
#ifdef SetupIconFileParam
SetupIconFile={#SetupIconFileParam}
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "setupmfa"; Description: "Install/verify MFA runtime now (online download)"; GroupDescription: "Optional setup"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#ReleaseFileExcludes}"
; Optional external override for rapid setup_mfa.bat testing:
; If setup_mfa.bat is placed next to Setup.exe, use it to overwrite bundled one.
Source: "{src}\setup_mfa.bat"; DestDir: "{app}"; DestName: "setup_mfa.bat"; Flags: external skipifsourcedoesntexist ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\{#MyAppName} Startup Diagnose"; Filename: "{app}\startup_diagnose.bat"; Check: FileExists(ExpandConstant('{app}\startup_diagnose.bat'))
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.env"
Type: filesandordirs; Name: "{app}\micromamba"
Type: filesandordirs; Name: "{app}\.mfa_root_ascii"
; Runtime fallback roots that can be used by setup_mfa/UI auto-repair.
Type: filesandordirs; Name: "{localappdata}\UTAU_Auto_OTO\.env"
Type: filesandordirs; Name: "{localappdata}\UTAU_Auto_OTO\micromamba"
Type: filesandordirs; Name: "{localappdata}\UTAU_Auto_OTO\.mfa_root_ascii"
Type: filesandordirs; Name: "{localappdata}\UTAU_Auto_OTO_v3\.env"
Type: filesandordirs; Name: "{localappdata}\UTAU_Auto_OTO_v3\micromamba"
Type: filesandordirs; Name: "{localappdata}\UTAU_Auto_OTO_v3\.mfa_root_ascii"

[Run]
Filename: "{app}\setup_mfa.bat"; Description: "Install/verify MFA runtime"; Flags: postinstall shellexec waituntilterminated skipifsilent; Tasks: setupmfa; Check: FileExists(ExpandConstant('{app}\setup_mfa.bat'))
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: postinstall nowait skipifsilent

[Code]
function HasRequiredVCRuntime(): Boolean;
var
  HasMsvcp140: Boolean;
  HasMsvcp140_1: Boolean;
  HasVcruntime140: Boolean;
  HasVcruntime140_1: Boolean;
begin
  HasMsvcp140 := FileExists(ExpandConstant('{app}\msvcp140.dll')) or FileExists(ExpandConstant('{sys}\msvcp140.dll'));
  HasMsvcp140_1 := FileExists(ExpandConstant('{app}\msvcp140_1.dll')) or FileExists(ExpandConstant('{sys}\msvcp140_1.dll'));
  HasVcruntime140 := FileExists(ExpandConstant('{app}\vcruntime140.dll')) or FileExists(ExpandConstant('{sys}\vcruntime140.dll'));
  HasVcruntime140_1 := FileExists(ExpandConstant('{app}\vcruntime140_1.dll')) or FileExists(ExpandConstant('{sys}\vcruntime140_1.dll'));
  Result := HasMsvcp140 and HasMsvcp140_1 and HasVcruntime140 and HasVcruntime140_1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    exit;

  if not FileExists(ExpandConstant('{app}\{#MyAppExeName}')) then begin
    MsgBox(
      '설치 후 실행 파일이 감지되지 않았습니다.'#13#10#13#10 +
      '경로: ' + ExpandConstant('{app}\{#MyAppExeName}') + #13#10#13#10 +
      '보안 프로그램(Defender/백신) 격리 여부를 확인한 뒤 재설치해 주세요.',
      mbCriticalError, MB_OK
    );
    exit;
  end;

  if not HasRequiredVCRuntime() then begin
    MsgBox(
      'Microsoft Visual C++ 2015-2022 (x64) 런타임이 감지되지 않았습니다.'#13#10 +
      '실행이 되지 않으면 VC++ 런타임 설치 후 다시 실행해 주세요.',
      mbInformation, MB_OK
    );
  end;
end;
