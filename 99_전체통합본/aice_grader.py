import pandas as pd
import numpy as np
import time

def check_nulls(df):
    """데이터프레임의 결측치가 모두 처리되었는지 확인합니다."""
    if not isinstance(df, pd.DataFrame):
        print("❌ [오류] pandas DataFrame 객체를 전달해주세요.")
        return
        
    null_count = df.isnull().sum().sum()
    if null_count == 0:
        print("✅ [정답] 완벽합니다! 결측치가 모두 처리되었습니다.")
    else:
        print(f"❌ [오답] 아직 결측치가 {null_count}개 남아있습니다. fillna() 또는 dropna()를 확인하세요.")

def check_encoding(df):
    """데이터프레임 내에 문자열(object) 타입이 모두 수치형으로 변환되었는지 확인합니다."""
    if not isinstance(df, pd.DataFrame):
        print("❌ [오류] pandas DataFrame 객체를 전달해주세요.")
        return
        
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) == 0:
        print("✅ [정답] 훌륭합니다! 모든 데이터가 수치형으로 인코딩되었습니다.")
    else:
        print(f"❌ [오답] 아직 인코딩되지 않은 컬럼이 있습니다: {list(object_cols)}. get_dummies()를 적용하세요.")

def check_shape(X_train, X_test):
    """Train과 Test 데이터의 컬럼(특성) 개수가 동일한지 확인합니다. (Shape mismatch 방지)"""
    if X_train.shape[1] == X_test.shape[1]:
        print(f"✅ [정답] Train과 Test의 컬럼 수가 {X_train.shape[1]}개로 일치합니다.")
    else:
        print(f"❌ [오답] Train 컬럼 수({X_train.shape[1]})와 Test 컬럼 수({X_test.shape[1]})가 다릅니다. 인코딩 과정에서 데이터 누수가 없는지 확인하세요.")

def check_score(score, threshold=0.7):
    """모델의 평가 점수(정확도, R2 등)가 목표 임계값을 넘겼는지 확인합니다."""
    if score >= threshold:
        print(f"✅ [합격] 점수가 {score:.4f}로 목표치({threshold})를 넘겼습니다! 모델 성능이 좋습니다.")
    else:
        print(f"⚠️ [주의] 점수가 {score:.4f}로 목표치({threshold})보다 낮습니다. 파라미터 튜닝이나 전처리를 다시 확인해보세요.")


class MockExamGrader:
    """모의고사를 풀면서 문제별 배점을 직접 매겨 100점 만점으로 자가 채점하는 도구.

    사용 예시:
        grader = MockExamGrader("모의고사01_Titanic")
        grader.check(1, points=6, is_correct=True)
        grader.check(2, points=6, is_correct=False, note="value_counts() 옵션 누락")
        grader.report()
    """

    def __init__(self, exam_name="모의고사"):
        self.exam_name = exam_name
        self.results = []

    def check(self, question_no, points, is_correct, note=""):
        self.results.append((question_no, points, bool(is_correct), note))
        status = "✅" if is_correct else "❌"
        earned = points if is_correct else 0
        msg = f"{status} 문제 {question_no} ({earned}/{points}점)"
        if note:
            msg += f" - {note}"
        print(msg)

    def report(self):
        if not self.results:
            print("아직 채점한 문제가 없습니다. check()를 먼저 호출하세요.")
            return
        earned_total = sum(p for _, p, c, _ in self.results if c)
        max_total = sum(p for _, p, _, _ in self.results)
        pct = (earned_total / max_total * 100) if max_total else 0
        print(f"\n=== {self.exam_name} 채점 결과 ===")
        print(f"획득 점수: {earned_total} / {max_total}  ({pct:.1f}%)")
        if pct >= 80:
            print("결과: 합격 기준(80점) 통과! 실전 감각이 잘 잡혀 있습니다.")
        else:
            wrong = [q for q, p, c, n in self.results if not c]
            print(f"결과: 합격 기준(80점) 미달. 틀린 문제: {wrong}")
            print("해당 문제 번호의 이론을 PART 3 가이드에서 다시 확인해보세요.")


class ExamTimer:
    """모의고사를 실제 시험처럼 시간 제한을 두고 풀 때 쓰는 타이머.

    사용 예시 (모의고사 노트북 맨 처음 셀에서):
        timer = ExamTimer(minutes=90)
        timer.start()

        # ... 문제를 풀다가 중간에 언제든 ...
        timer.check()

        # 다 풀고 나서 (채점 가이드 직전)
        timer.finish()
    """

    def __init__(self, minutes=90):
        self.minutes = minutes
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        started_at = time.strftime('%H:%M:%S', time.localtime(self.start_time))
        print(f"⏱️ 타이머 시작! (시작 시각 {started_at}) 총 {self.minutes}분 안에 풀어보세요.")

    def check(self):
        if self.start_time is None:
            print("⚠️ 아직 타이머를 시작하지 않았습니다. timer.start()를 먼저 호출하세요.")
            return
        elapsed = (time.time() - self.start_time) / 60
        remaining = self.minutes - elapsed
        if remaining > 0:
            print(f"⏳ 경과 {elapsed:.1f}분 / 남은 시간 {remaining:.1f}분")
            if remaining <= 10:
                print("⚠️ 10분 이내로 남았습니다! 남은 문제부터 빠르게 채워나가세요.")
        else:
            print(f"⏰ 시간 종료! 경과 {elapsed:.1f}분 (제한시간 {self.minutes}분 초과)")
            print("실전이었다면 여기서 제출입니다. 어디까지 풀었는지 점검해보세요.")

    def finish(self):
        if self.start_time is None:
            print("⚠️ 타이머가 시작되지 않아 소요 시간을 계산할 수 없습니다.")
            return
        elapsed = (time.time() - self.start_time) / 60
        print(f"\n=== 최종 소요 시간: {elapsed:.1f}분 / {self.minutes}분 ===")
        if elapsed <= self.minutes:
            print(f"✅ 제한시간 내 완료! ({self.minutes - elapsed:.1f}분을 남기고 끝냈습니다)")
        else:
            print(f"⚠️ 제한시간을 {elapsed - self.minutes:.1f}분 초과했습니다. "
                  f"프리미엄 가이드북 19장(시간 배분 전략)을 참고해 전처리 단계 속도를 높여보세요.")


def compare_value(actual, expected, tolerance=1e-3, label="값"):
    """실제 계산값이 기대값과 (허용오차 내에서) 같은지 확인합니다. 숫자/문자열 모두 지원."""
    try:
        is_close = abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        is_close = (actual == expected)
    if is_close:
        print(f"✅ [정답] {label}: {actual} (기대값 {expected}과 일치)")
    else:
        print(f"❌ [오답] {label}: {actual} (기대값 {expected}과 다름)")
    return is_close


if __name__ == '__main__':
    print("AICE Auto-Grader 모듈이 정상적으로 로드되었습니다.")
