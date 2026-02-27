import os
import shutil
import subprocess
import sys

def main():
    print("📦 [1/4] 파이프라인 패키징 시작: 필요한 빌드 모듈 확인 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "customtkinter", "textgrid", "numpy"])

    print("🚀 [2/4] PyInstaller 모듈 로딩 중...")
    import PyInstaller.__main__
    import customtkinter
    
    ctk_path = os.path.dirname(customtkinter.__file__)
    
    print("⚙️ [3/4] UTAU_Auto_OTO.exe 컴파일 빌드 시작 (수 분이 소요될 수 있습니다)...")
    
    PyInstaller.__main__.run([
        'main.py',                       # 메인 진입점
        '--name=UTAU_Auto_OTO',          # 생성될 exe 이름
        '--windowed',                    # 콘솔 창 숨기기 (GUI 전용)
        '--onefile',                     # 단일 실행 파일로 압축
        '--noconfirm',                   # 기존 빌드 덮어쓰기
        '--clean',                       # 캐시 정리
        f'--add-data={ctk_path};customtkinter/', # CustomTkinter 인터페이스 에셋 포함
        '--hidden-import=textgrid',      # 명시적 textgrid 포함
        '--hidden-import=customtkinter', # 명시적 ctk 포함
    ])

    print("📁 [4/4] 배포용 폴더 조립 중...")
    release_dir = "UTAU_Auto_OTO_Release"
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)

    # 1. 단일 실행 파일 복사
    exe_path = os.path.join("dist", "UTAU_Auto_OTO.exe")
    if os.path.exists(exe_path):
        shutil.copy(exe_path, release_dir)
        print(f"   -> {exe_path} 복사 완료")
    else:
        print("   ❌ 빌드된 exe 파일을 찾을 수 없습니다!")

    # 2. MFA 설치용 스크립트 복사
    setup_path = "setup_mfa.bat"
    if os.path.exists(setup_path):
        shutil.copy(setup_path, release_dir)
        print("   -> setup_mfa.bat 복사 완료")

    print(f"\n🎉 [완료] 모든 빌드 과정이 성공했습니다! '{release_dir}' 폴더를 확인해 주세요.")

if __name__ == '__main__':
    main()
