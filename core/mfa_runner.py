"""
MFA (Montreal Forced Aligner) 실행 모듈
- 로컬 또는 포터블 Conda 환경에서 MFA 실행
- 실시간 로그 스트리밍
"""

import os
import sys
import subprocess
import re
import logging
import shutil

logger = logging.getLogger(__name__)

def _get_conda_env(mfa_path):
    """
    Windows 환경에서 Conda 활성화 없이 mfa.exe를 직접 호출할 때 
    DLL 로드 에러(코드 3228369023)가 발생하는 것을 막기 위해 환경 변수에 PATH를 주입합니다.
    """
    env = os.environ.copy()
    if sys.platform == 'win32' and mfa_path and 'Scripts' in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        new_paths = [
            env_dir,
            os.path.join(env_dir, 'Library', 'mingw-w64', 'bin'),
            os.path.join(env_dir, 'Library', 'usr', 'bin'),
            os.path.join(env_dir, 'Library', 'bin'),
            os.path.join(env_dir, 'Scripts'),
            os.path.join(env_dir, 'bin'),
        ]
        current_path = env.get('PATH', '')
        env['PATH'] = os.pathsep.join(new_paths) + os.pathsep + current_path
        env['CONDA_PREFIX'] = env_dir
    return env


def find_mfa_executable():
    """
    시스템에 설치된 MFA 실행 파일을 탐색합니다.
    포터블 환경 -> Conda 환경 -> 시스템 PATH 순서로 검색합니다.
    
    Returns:
        MFA 실행 파일 경로 또는 None
    """
    # 1. 포터블 환경 (프로그램 폴더 내 .env/)
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    portable_paths = [
        os.path.join(app_dir, '.env', 'Scripts', 'mfa.exe'),
        os.path.join(app_dir, '.env', 'bin', 'mfa'),
        os.path.join(app_dir, 'env', 'Scripts', 'mfa.exe'),
    ]
    public_root = os.environ.get('PUBLIC', r'C:\Users\Public')
    portable_paths.extend([
        os.path.join(public_root, 'UTAU_Auto_OTO_v3', '.env', 'Scripts', 'mfa.exe'),
        os.path.join(public_root, 'UTAU_Auto_OTO_v3', '.env', 'bin', 'mfa'),
    ])
    for p in portable_paths:
        if os.path.exists(p):
            logger.info(f"포터블 MFA 발견: {p}")
            return p

    # 2. 시스템 PATH
    mfa_path = shutil.which('mfa')
    if mfa_path:
        logger.info(f"시스템 MFA 발견: {mfa_path}")
        return mfa_path

    # 3. Conda 환경 기본 경로
    conda_paths = [
        os.path.expanduser('~/miniconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/anaconda3/envs/aligner/Scripts/mfa.exe'),
        os.path.expanduser('~/miniconda3/Scripts/mfa.exe'),
    ]
    for p in conda_paths:
        if os.path.exists(p):
            logger.info(f"Conda MFA 발견: {p}")
            return p

    return None


def check_mfa_model(mfa_path, language='korean'):
    """
    MFA 음향 모델이 다운로드되어 있는지 확인합니다.
    
    Args:
        mfa_path: MFA 실행 파일 경로
        language: 'korean' 또는 'japanese'
    
    Returns:
        (설치 여부: bool, 메시지: str)
    """
    if not mfa_path:
        return False, "MFA 실행 파일을 찾을 수 없습니다."

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = '일본어' if language == 'japanese' else '한국어'

    try:
        env = _get_conda_env(mfa_path)
        result = subprocess.run(
            [mfa_path, 'model', 'list', 'acoustic'],
            capture_output=True, text=True, timeout=30, env=env
        )
        if model_name in result.stdout:
            return True, f"{lang_label} MFA 모델이 설치되어 있습니다."
        else:
            return False, f"{lang_label} MFA 모델이 설치되어 있지 않습니다. 다운로드가 필요합니다."
    except Exception as e:
        return False, f"MFA 모델 확인 실패: {e}"


def ensure_korean_support(mfa_path, callback=None):
    """
    MFA 실행 전, 한국어 처리에 필요한 eunjeon, jamo 모듈이 설치되어 있는지 확인하고
    설치되어 있지 않다면 즉시 설치한 후 MFA 소스코드를 패치합니다.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or 'Scripts' not in mfa_path:
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = os.path.join(env_dir, 'python.exe')
    
    if not os.path.exists(python_exe):
        return True

    # Check if eunjeon and jamo are installed
    check_cmd = [python_exe, '-c', 'import eunjeon; import jamo']
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Already installed, but ensure patch is applied just in case
            patch_mfa_korean_support(mfa_path, callback)
            return True
            
        log("📦 MFA 한국어 구문 분석용 필수 라이브러리(eunjeon, jamo) 설치/확인 중...")
        pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
        
        # Determine if we should use conda run or pip directly
        install_cmd = None
        system_conda = shutil.which('conda')
        if system_conda:
            install_cmd = [system_conda, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo']
        elif os.path.exists(pip_exe):
            install_cmd = [pip_exe, 'install', 'eunjeon', 'jamo']
            
        if install_cmd:
            log(f"   -> 실행 명령어: {' '.join(install_cmd)}")
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                 log(f"   ⚠️ 설치 중 에러: {result.stderr}")
            log("✅ 설치 시도 완료! MFA 한국어 연동 패치를 진행합니다...")
            patch_mfa_korean_support(mfa_path, callback)
        return True
    except Exception as e:
        log(f"⚠️ 의존성 자동 확인/설치 중 오류 발생: {e}")
        return False


def ensure_japanese_support(mfa_path, callback=None):
    """
    MFA 일본어 정렬에 필요한 spacy/sudachipy/sudachidict-core가 있는지 확인하고,
    누락 시 자동 설치를 시도합니다.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path or 'Scripts' not in mfa_path:
        return True

    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = os.path.join(env_dir, 'python.exe')
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')

    if not os.path.exists(python_exe):
        return True

    check_cmd = [python_exe, '-c', 'import spacy; import sudachipy; import sudachidict_core']
    try:
        env = _get_conda_env(mfa_path)
        result = subprocess.run(check_cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            return True

        log("📦 MFA 일본어 토크나이저 의존성(spacy, sudachipy, sudachidict-core) 설치/확인 중...")

        install_cmd = None
        if os.path.exists(conda_exe):
            install_cmd = [
                conda_exe, 'install', '-y', '-p', env_dir,
                '-c', 'conda-forge', '--override-channels',
                'spacy', 'sudachipy', 'sudachidict-core'
            ]
        else:
            system_conda = shutil.which('conda')
            if system_conda:
                install_cmd = [
                    system_conda, 'install', '-y', '-p', env_dir,
                    '-c', 'conda-forge', '--override-channels',
                    'spacy', 'sudachipy', 'sudachidict-core'
                ]
            elif os.path.exists(pip_exe):
                install_cmd = [pip_exe, 'install', 'spacy', 'sudachipy', 'sudachidict-core']

        if not install_cmd:
            log("⚠️ 일본어 의존성 자동 설치 경로를 찾지 못했습니다.")
            return False

        log(f"   -> 실행 명령어: {' '.join(install_cmd)}")
        install_result = subprocess.run(install_cmd, capture_output=True, text=True, env=env)
        if install_result.returncode != 0:
            if install_result.stderr:
                log(f"   ⚠️ 설치 stderr: {install_result.stderr[:500]}")
            if install_result.stdout:
                log(f"   ⚠️ 설치 stdout: {install_result.stdout[:500]}")
            if os.path.exists(pip_exe):
                pip_cmd = [pip_exe, 'install', 'spacy', 'sudachipy', 'sudachidict-core']
                log(f"   -> 대체 설치 명령어(pip): {' '.join(pip_cmd)}")
                pip_result = subprocess.run(pip_cmd, capture_output=True, text=True, env=env)
                if pip_result.returncode != 0:
                    if pip_result.stderr:
                        log(f"   ⚠️ pip stderr: {pip_result.stderr[:500]}")
                    if pip_result.stdout:
                        log(f"   ⚠️ pip stdout: {pip_result.stdout[:500]}")
                    return False
            else:
                return False

        verify = subprocess.run(check_cmd, capture_output=True, text=True, env=env)
        if verify.returncode == 0:
            log("✅ 일본어 토크나이저 의존성 설치 확인 완료")
            return True

        log("⚠️ 일본어 의존성 설치 후에도 import 검증에 실패했습니다.")
        if verify.stderr:
            log(f"   상세 stderr: {verify.stderr[:500]}")
        return False
    except Exception as e:
        log(f"⚠️ 일본어 의존성 자동 확인/설치 중 오류 발생: {e}")
        return False


def download_mfa_model(mfa_path, language='korean', callback=None):
    """한국어/일본어 MFA 음향 모델을 다운로드합니다."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        log("❌ MFA 실행 파일을 찾을 수 없습니다.")
        return False

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = '일본어' if language == 'japanese' else '한국어'

    # 한국어 전용 패치 (일본어는 불필요)
    if language == 'korean':
        ensure_korean_support(mfa_path, callback)

    log(f"📥 {lang_label} MFA 모델 다운로드 중... (최초 1회만 필요)")
    try:
        env = _get_conda_env(mfa_path)
        process = subprocess.Popen(
            [mfa_path, 'model', 'download', 'acoustic', model_name, '--ignore_cache'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', env=env
        )
        if process.stdout:
            for line in process.stdout:
                log(line.strip())
        process.wait()
        if process.returncode == 0:
            log(f"✅ {lang_label} MFA 모델 다운로드 완료!")
            return True
        else:
            log(f"❌ 모델 다운로드 실패 (코드: {process.returncode})")
            return False
    except Exception as e:
        log(f"❌ 모델 다운로드 에러: {e}")
        return False


def run_mfa_align(mfa_path, wav_folder, dict_path, output_folder, language='korean', callback=None):
    """
    MFA 음성-텍스트 강제 정렬을 실행합니다.
    
    Args:
        mfa_path: MFA 실행 파일 경로
        wav_folder: WAV + Lab 파일이 있는 폴더
        dict_path: 사전 파일 경로
        output_folder: TextGrid 출력 폴더
        language: 'korean' 또는 'japanese'
        callback: 실시간 로그 콜백 함수
    
    Returns:
        (성공 여부: bool, 에러 메시지: str)
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        return False, "MFA 실행 파일을 찾을 수 없습니다."

    if not os.path.exists(wav_folder):
        return False, f"WAV 폴더를 찾을 수 없습니다: {wav_folder}"

    if not os.path.exists(dict_path):
        return False, f"사전 파일을 찾을 수 없습니다: {dict_path}"

    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = '일본어' if language == 'japanese' else '한국어'

    log(f"🔍 MFA 정렬 전제 조건 확인 중... ({lang_label})")
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            log("⚠️ 한국어 지원 라이브러리 확인 중 문제가 발생했습니다. 계속 시도합니다.")
    elif language == 'japanese':
        if not ensure_japanese_support(mfa_path, callback):
            log("⚠️ 일본어 지원 라이브러리 확인 중 문제가 발생했습니다. 정렬은 계속 시도합니다.")

    os.makedirs(output_folder, exist_ok=True)
    log(f"📂 출력 폴더 생성/확인: {output_folder}")

    cmd = [
        mfa_path, 'align',
        wav_folder, dict_path, model_name, output_folder,
        '--clean', '--fine_tune', '--textgrid_cleanup',
        '--beam', '1000', '--retry_beam', '4000',
        '--num_jobs', '1'
    ]

    log(f"🚀 MFA 정렬 시작...")
    log(f"   WAV 폴더: {wav_folder}")
    log(f"   사전 파일: {dict_path}")
    log(f"   출력 폴더: {output_folder}")
    log("   정밀 정렬 모드: fine_tune + textgrid_cleanup + 확장 beam")
    log(f"   ⏳ PC 사양에 따라 5~15분 정도 소요될 수 있습니다...")

    try:
        env = _get_conda_env(mfa_path)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        if process.stdout:
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    log(stripped)
        process.wait()

        if process.returncode == 0:
            log("✅ MFA 정렬이 성공적으로 완료되었습니다!")
            return True, ""
        else:
            err = f"MFA 정렬 실패 (종료 코드: {process.returncode})"
            log(f"❌ {err}")
            return False, err

    except FileNotFoundError:
        err = "MFA 실행 파일을 찾을 수 없습니다. MFA가 올바르게 설치되어 있는지 확인해 주세요."
        log(f"❌ {err}")
        return False, err
    except Exception as e:
        err = f"MFA 실행 중 예기치 않은 에러 발생: {e}"
        log(f"❌ {err}")
        return False, err

def patch_mfa_korean_support(mfa_path, callback=None):
    """
    Windows 환경에서 python-mecab-ko는 C++ 빌드 툴이 없어 설치가 실패합니다.
    대신 윈도우용 사전 컴파일된 eunjeon(mecab-ko 포크)를 설치한 뒤, 
    MFA 내부 소스코드(spacy.py, korean.py)가 eunjeon을 참조하도록 강제로 패치합니다.
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if sys.platform != 'win32' or not mfa_path or 'Scripts' not in mfa_path:
        return True
        
    try:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        site_packages = os.path.join(env_dir, 'Lib', 'site-packages', 'montreal_forced_aligner')
        
        spacy_py = os.path.join(site_packages, 'tokenization', 'spacy.py')
        korean_py = os.path.join(site_packages, 'tokenization', 'korean.py')
        
        # 1. spacy.py 패치: 'mecab' 대신 'eunjeon'을 체크하도록 수정
        if os.path.exists(spacy_py):
            with open(spacy_py, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # (1) 에러 메시지 수정 (eunjeon 안내 포함)
            if 'pip install python-mecab-ko jamo' in content:
                content = content.replace("pip install python-mecab-ko jamo", "pip install eunjeon jamo")
            
            # (2) 가용성 체크 로직 수정: 'from mecab import' 나 'import mecab'을 eunjeon으로 우회
            if "import mecab" in content and "import eunjeon" not in content:
                content = content.replace("import mecab", "import eunjeon")
                
            with open(spacy_py, 'w', encoding='utf-8') as f:
                f.write(content)
            log("   [Patch] spacy.py 가용성 체크 수정 완료")

        # 2. korean.py 패치: KO_AVAILABLE를 True로 만들고 Eunjeon 래퍼 주입
        if os.path.exists(korean_py):
            with open(korean_py, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # (1) KO_AVAILABLE 결정 로직 수정 (eunjeon이 있으면 True가 되도록)
            # 중복 패치 방지 및 손상된 파일 복구 로직 추가
            if 'EunjeonWrapper' not in content:
                # 이미 잘못된 패치가 되어있는 경우 (중첩 try) 복구 시도
                content = re.sub(r'try:\s+try:\s+from mecab import MeCab', 'from mecab import MeCab', content)
                content = re.sub(r'try:\s+from mecab import MeCab', 'from mecab import MeCab', content)
                
                # 정석적인 4분할 시퀀스로 교체 (정확한 인덴트 유지)
                if '    from mecab import MeCab' in content:
                    content = content.replace(
                        '    from mecab import MeCab', 
                        '    try:\n        from mecab import MeCab\n    except:\n        from eunjeon import Mecab as MeCab'
                    )
                elif 'from mecab import MeCab' in content:
                    # 인덴트가 없는 경우 (가능성은 낮지만 방어용)
                    content = content.replace(
                        'from mecab import MeCab', 
                        'try:\n    from mecab import MeCab\nexcept:\n    from eunjeon import Mecab as MeCab'
                    )
                
            wrapper_code = '''
class EunjeonNode:
    def __init__(self, surface, pos):
        self.surface = surface
        self.pos = pos

class EunjeonWrapper:
    def __init__(self):
        from eunjeon import Mecab
        self.mecab = Mecab()
        
    def parse(self, text):
        return [EunjeonNode(w, p) for w, p in self.mecab.pos(text)]
'''
            if 'class EunjeonWrapper' not in content:
                # Add wrapper class after imports
                content = content.replace('class KoreanTokenizer:', wrapper_code + '\nclass KoreanTokenizer:')
                # Replace the tokenizer instantiation
                if 'self.tokenizer = MeCab()' in content:
                    content = content.replace("self.tokenizer = MeCab()", "self.tokenizer = EunjeonWrapper()")
                
            with open(korean_py, 'w', encoding='utf-8') as f:
                f.write(content)
            log("   [Patch] korean.py 참조 수정 (KO_AVAILABLE 및 Eunjeon 래퍼) 완료")
                
        return True
    except Exception as e:
        log(f"⚠️ MFA 한국어 패치 중 오류 발생: {e}")
        return False

