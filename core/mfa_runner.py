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

ALERT_MSVC_REQUIRED = "__ALERT__MSVC_REQUIRED__"
ALERT_MFA_PERMISSION_DENIED = "__ALERT__MFA_PERMISSION_DENIED__"
MSVC_REQUIRED_TEXT = "microsoft visual c++ 14.0 or greater is required"


def _stderr_has_msvc_requirement(text):
    if not text:
        return False
    lowered = text.lower()
    return (
        MSVC_REQUIRED_TEXT in lowered
        or "visual-cpp-build-tools" in lowered
        or "microsoft c++ build tools" in lowered
    )


def _emit_msvc_required_notice(callback, log_fn):
    if callback:
        callback(ALERT_MSVC_REQUIRED)
    log_fn("⚠ Microsoft Visual C++ 14.0+ (C++ Build Tools)가 필요합니다.")
    log_fn("   설치 링크: https://visualstudio.microsoft.com/visual-cpp-build-tools/")


def _preflight_compute_mfcc(mfa_path, callback=None):
    """MFA 정렬 시작 전에 compute-mfcc-feats 실행 가능 여부를 점검합니다."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    if not mfa_path:
        return False, "MFA 실행 파일 경로가 비어 있습니다."

    env = _get_conda_env(mfa_path)
    candidates = []
    if sys.platform == 'win32' and 'Scripts' in mfa_path:
        env_dir = os.path.dirname(os.path.dirname(mfa_path))
        candidates.append(os.path.join(env_dir, 'Library', 'bin', 'compute-mfcc-feats.exe'))
        candidates.append('compute-mfcc-feats.exe')
    candidates.append('compute-mfcc-feats')

    last_not_found = None
    for candidate in candidates:
        try:
            # Windows + Python 3.13 조합에서는 확장자 없는 실행명 검색이 실패할 수 있다.
            subprocess.run(
                [candidate, '--help'],
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )
            return True, ""
        except FileNotFoundError as e:
            last_not_found = e
            continue
        except PermissionError as e:
            if callback:
                callback(ALERT_MFA_PERMISSION_DENIED)
            err = (
                "compute-mfcc-feats 실행 권한이 없어 MFA 정렬을 시작할 수 없습니다. "
                "(WinError 5: Access denied)"
            )
            log(f"❌ {err}")
            log("   보안 프로그램/권한 정책/파일 차단 여부를 확인해 주세요.")
            return False, f"{err}: {e}"
        except Exception as e:
            err = f"compute-mfcc-feats 사전 점검 중 오류: {e}"
            log(f"❌ {err}")
            return False, err

    err = "compute-mfcc-feats를 찾지 못했습니다. MFA 환경이 손상되었을 수 있습니다."
    log(f"❌ {err}")
    if last_not_found:
        return False, f"{err}: {last_not_found}"
    return False, err

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
    Ensure Korean MFA tokenizer dependencies are available:
    - eunjeon
    - jamo
    """
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        return False
    if 'Scripts' not in mfa_path:
        # System MFA path; skip env-local auto install here.
        return True
    env_dir = os.path.dirname(os.path.dirname(mfa_path))
    python_exe = os.path.join(env_dir, 'python.exe')
    pip_exe = os.path.join(env_dir, 'Scripts', 'pip.exe')
    if not os.path.exists(python_exe):
        return False
    check_cmd = [python_exe, '-c', 'import eunjeon; import jamo']
    try:
        env = _get_conda_env(mfa_path)

        def _looks_like_pyexpat_dll_issue(msg):
            s = (msg or '').lower()
            return ('pyexpat' in s and 'dll load failed' in s) or ('libexpat' in s and 'not found' in s)

        def _try_repair_pyexpat():
            conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
            cmds = []
            if os.path.exists(conda_exe):
                cmds.append([conda_exe, 'install', '-y', '-p', env_dir, 'libexpat'])
            system_conda = shutil.which('conda')
            if system_conda:
                cmds.append([system_conda, 'install', '-y', '-p', env_dir, 'libexpat'])
            for cmd in cmds:
                log(f"   -> repair cmd: {' '.join(cmd)}")
                res = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if res.returncode == 0:
                    return True
            return False

        def _check_imports():
            res = subprocess.run(check_cmd, capture_output=True, text=True, env=env)
            if res.returncode == 0:
                return True, ''
            detail = (res.stderr or res.stdout or '').strip()
            return False, detail
        ok, detail = _check_imports()
        if (not ok) and _looks_like_pyexpat_dll_issue(detail):
            log('[MFA] Detected pyexpat/libexpat DLL issue; trying repair...')
            if _try_repair_pyexpat():
                ok, detail = _check_imports()
        if ok:
            patch_mfa_korean_support(mfa_path, callback)
            return True
        log('[MFA] Installing Korean tokenizer deps: eunjeon, jamo')
        install_cmds = [
            [python_exe, '-m', 'pip', 'install', '--upgrade', 'eunjeon', 'jamo'],
        ]
        if os.path.exists(pip_exe):
            install_cmds.append([pip_exe, 'install', '--upgrade', 'eunjeon', 'jamo'])
        system_conda = shutil.which('conda')
        if system_conda:
            install_cmds.append([
                system_conda, 'run', '-p', env_dir, 'python', '-m', 'pip', 'install',
                '--upgrade', 'eunjeon', 'jamo'
            ])
        last_err = detail
        for install_cmd in install_cmds:
            log(f"   -> cmd: {' '.join(install_cmd)}")
            result = subprocess.run(install_cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                err_txt = (result.stderr or result.stdout or '').strip()
                if err_txt:
                    log(f"   [warn] install failed: {err_txt[:500]}")
                if _stderr_has_msvc_requirement(result.stderr):
                    _emit_msvc_required_notice(callback, log)
                last_err = err_txt or last_err
                continue
            ok, detail = _check_imports()
            if (not ok) and _looks_like_pyexpat_dll_issue(detail):
                log('[MFA] Detected pyexpat/libexpat DLL issue after install; trying repair...')
                if _try_repair_pyexpat():
                    ok, detail = _check_imports()
            if ok:
                log('[MFA] Korean tokenizer deps are ready')
                patch_mfa_korean_support(mfa_path, callback)
                return True
            if detail:
                log(f"   [warn] import check failed after install: {detail[:500]}")
                last_err = detail
        log('[MFA] Failed to prepare Korean tokenizer deps (eunjeon, jamo)')
        if last_err:
            log(f"   last error: {last_err[:500]}")
        return False
    except Exception as e:
        log(f"[MFA] Korean dependency setup error: {e}")
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
    """Download MFA acoustic model for selected language."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        log('MFA executable not found.')
        return False
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            log('Failed to prepare Korean dependencies (eunjeon, jamo).')
            return False
    log(f'Downloading {lang_label} MFA model...')
    try:
        env = _get_conda_env(mfa_path)
        process = subprocess.Popen(
            [mfa_path, 'model', 'download', 'acoustic', model_name, '--ignore_cache'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if line:
                    log(line)
        process.wait()
        if process.returncode == 0:
            log(f'{lang_label} MFA model download completed.')
            return True
        log(f'Model download failed (code: {process.returncode})')
        return False
    except Exception as e:
        log(f'Model download error: {e}')
        return False
def run_mfa_align(mfa_path, wav_folder, dict_path, output_folder, language='korean', callback=None):
    """Run MFA forced alignment."""
    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)
    if not mfa_path:
        return False, 'MFA executable not found.'
    if not os.path.exists(wav_folder):
        return False, f'WAV folder not found: {wav_folder}'
    if not os.path.exists(dict_path):
        return False, f'Dictionary not found: {dict_path}'
    model_name = 'japanese_mfa' if language == 'japanese' else 'korean_mfa'
    lang_label = 'Japanese' if language == 'japanese' else 'Korean'
    log(f'Checking MFA prerequisites... ({lang_label})')
    if language == 'korean':
        if not ensure_korean_support(mfa_path, callback):
            err = 'Missing Korean tokenizer dependencies (eunjeon, jamo).'
            log(err)
            return False, err
    elif language == 'japanese':
        if not ensure_japanese_support(mfa_path, callback):
            err = 'Missing Japanese tokenizer dependencies.'
            log(err)
            return False, err
    os.makedirs(output_folder, exist_ok=True)
    ok, preflight_err = _preflight_compute_mfcc(mfa_path, callback=callback)
    if not ok:
        return False, preflight_err
    cmd = [
        mfa_path, 'align',
        wav_folder, dict_path, model_name, output_folder,
        '--clean', '--fine_tune', '--textgrid_cleanup',
        '--beam', '1000', '--retry_beam', '4000',
        '--num_jobs', '1',
    ]
    log('Starting MFA alignment...')
    try:
        env = _get_conda_env(mfa_path)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        tail_lines = []
        if process.stdout:
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    log(stripped)
                    tail_lines.append(stripped)
                    if len(tail_lines) > 120:
                        tail_lines.pop(0)
        process.wait()
        if process.returncode == 0:
            log('MFA alignment completed successfully.')
            return True, ''
        joined_tail = '\n'.join(tail_lines[-40:])
        lowered_tail = joined_tail.lower()
        if (
            'please install korean support' in lowered_tail
            or ('importerror' in lowered_tail and 'eunjeon' in lowered_tail and 'jamo' in lowered_tail)
        ):
            err = 'Korean dependencies (eunjeon, jamo) are missing in MFA env.'
            log(err)
            return False, err
        err = f'MFA alignment failed (code: {process.returncode})'
        if tail_lines:
            err += f' | tail: {tail_lines[-1][:180]}'
        log(err)
        return False, err
    except FileNotFoundError:
        err = 'MFA executable not found. Check MFA installation.'
        log(err)
        return False, err
    except Exception as e:
        err = f'Unexpected MFA error: {e}'
        log(err)
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





