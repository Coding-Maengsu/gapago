import {
  FileSearch, Brain, Target, Search, Database, Sparkles, FileText,
  Zap, BarChart3, Clock, MessageSquare,
  type LucideIcon,
} from 'lucide-react'

// ── 네비게이션 ──
export const NAV_LINKS = [
  { label: '사용법', href: '#workflow' },
  { label: '하는 일', href: '#service' },
  { label: '차별점', href: '#difference' },
  { label: '예시', href: '#examples' },
] as const

// ── ServiceSection 카드 데이터 ──
export type ServiceCard = { icon: LucideIcon; title: string; description: string }

export const SERVICE_CARDS: ServiceCard[] = [
  {
    icon: FileSearch,
    title: '논문 수집 & 분석',
    description: '관심 분야 키워드를 입력하면, 여러 글로벌 학술 데이터베이스에서 관련 논문을 자동으로 수집하고 관련성 순으로 정리합니다.',
  },
  {
    icon: Brain,
    title: 'AI 기반 연구 GAP 도출',
    description: '수집된 논문들의 한계점을 분석하고, 논문 간 교차 비교를 통해 아직 연구되지 않은 빈 영역을 찾아냅니다.',
  },
  {
    icon: Target,
    title: '연구 방향 제안',
    description: '도출된 연구 GAP을 바탕으로 새로운 연구 주제와 방향을 제안하고, 결과 검증을 통해 신뢰도 높은 결과를 보장합니다.',
  },
]

// ── WorkflowSection 스텝 데이터 ──
export type WorkflowStep = { step: string; icon: LucideIcon; title: string; description: string }

export const WORKFLOW_STEPS: WorkflowStep[] = [
  { step: '01', icon: Search, title: '키워드 입력', description: '연구하고 싶은 주제나 관심 분야의 키워드를 입력하세요. 예: "LLM 할루시네이션 탐지", "자율주행 안전성"' },
  { step: '02', icon: Database, title: '논문 자동 수집', description: '입력한 키워드를 기반으로 여러 학술 DB에서 관련 논문을 자동으로 검색하고 관련성 순으로 정리합니다.' },
  { step: '03', icon: Sparkles, title: '연구 GAP 분석', description: 'AI가 수집된 논문들의 한계점을 분석하고, 연구 주제별로 비어 있는 영역을 찾아냅니다.' },
  { step: '04', icon: FileText, title: '결과 리포트 확인', description: '도출된 연구 GAP과 추천 연구 방향이 담긴 리포트를 확인하고 추가 질문할 수 있습니다.' },
]

// ── FeaturesSection 기능 데이터 ──
export type FeatureItem = { icon: LucideIcon; title: string; description: string }

export const EXTRA_FEATURES: FeatureItem[] = [
  { icon: Zap, title: '빠른 자동 분석', description: '키워드 입력 후 몇 분 안에 논문 수집부터 연구 GAP 도출까지 전 과정을 자동 수행합니다.' },
  { icon: BarChart3, title: '구조화된 리포트', description: 'AI가 생성한 연구 축별로 GAP을 분류하고, 근거 논문과 함께 체계적인 리포트를 제공합니다.' },
  { icon: Clock, title: '분석 이력 & 대화형 탐색', description: '이전 분석 결과를 저장하고 다시 확인할 수 있으며, 추가 질문으로 심층 탐색이 가능합니다.' },
  { icon: MessageSquare, title: '키워드 자동 확장', description: '입력한 연구 주제를 AI가 분석하여 검색 효율을 높이기 위한 관련 키워드를 자동으로 확장합니다.' },
]

// ── DifferenceSection 비교 데이터 ──
export type ComparisonRow = {
  feature: string
  gapago: boolean
  gapagoDetail: string
  existing: boolean
  existingDetail: string
}

export const COMPARISON_DATA: ComparisonRow[] = [
  {
    feature: '논문 연구 GAP 도출',
    gapago: true,
    gapagoDetail: 'AI가 한계점 추출부터 연구 GAP 분석까지 자동 수행',
    existing: false,
    existingDetail: '연구 GAP 도출 기능 없음 (요약만 제공)',
  },
  {
    feature: '다중 논문 교차 분석',
    gapago: true,
    gapagoDetail: '수십 편 논문의 한계점을 교차 비교하여 공통 패턴 도출',
    existing: false,
    existingDetail: '개별 논문 단위 분석만 가능',
  },
  {
    feature: '연구 방향 제안',
    gapago: true,
    gapagoDetail: '연구 GAP 기반 구체적 연구 주제 및 방향 제안 + 결과 검증',
    existing: false,
    existingDetail: '연구 방향 제안 기능 없음',
  },
  {
    feature: '구조화된 리포트',
    gapago: true,
    gapagoDetail: '연구 축별 GAP 분류 + 근거 논문 + 연구 방향 포함 리포트',
    existing: false,
    existingDetail: '단순 텍스트 요약만 제공',
  },
  {
    feature: '논문 수집 자동화',
    gapago: true,
    gapagoDetail: '여러 학술 DB에서 병렬 검색 후 관련성 기반 다단계 선별',
    existing: true,
    existingDetail: '키워드 기반 검색 가능하나 정밀도 낮음',
  },
  {
    feature: '대화형 후속 탐색',
    gapago: true,
    gapagoDetail: '분석 결과에 대해 자연어로 추가 질문 및 심층 탐색 가능',
    existing: true,
    existingDetail: '기본적인 대화형 탐색 지원',
  },
]

// ── ExampleSection 예시 데이터 ──
export type ExampleGap = {
  title: string
  detail: string
  relatedPapers: number
}

export type ExampleCard = {
  query: string
  paperCount: number
  axes: string[]
  gaps: ExampleGap[]
}

export const EXAMPLES: ExampleCard[] = [
  {
    query: '자율주행 센서 퓨전 안전성 검증',
    paperCount: 38,
    axes: ['센서 퓨전', '시뮬레이션 검증', '엣지 케이스 대응'],
    gaps: [
      {
        title: '센서 퓨전 엣지 케이스 처리 부족',
        detail: '라이다-카메라 퓨전 환경에서 극단적 날씨/조명 조건의 엣지 케이스를 체계적으로 다룬 연구가 부족함',
        relatedPapers: 7,
      },
      {
        title: '악천후 시뮬레이션 검증 미비',
        detail: '폭우, 안개 등 악천후 환경에서의 자율주행 안전성을 시뮬레이션 기반으로 검증한 대규모 벤치마크가 없음',
        relatedPapers: 5,
      },
    ],
  },
  {
    query: 'LLM 할루시네이션 탐지 및 완화',
    paperCount: 52,
    axes: ['탐지 메트릭', '도메인 특화', '멀티모달 검증'],
    gaps: [
      {
        title: '도메인 특화 탐지 메트릭 부재',
        detail: '의료, 법률 등 전문 도메인에서의 할루시네이션을 정량적으로 측정할 수 있는 표준 메트릭이 확립되지 않음',
        relatedPapers: 12,
      },
      {
        title: '멀티모달 사실 검증 방법론 미흡',
        detail: '텍스트+이미지 입력에서 발생하는 교차 모달 할루시네이션에 대한 검증 프레임워크가 부족함',
        relatedPapers: 8,
      },
    ],
  },
  {
    query: 'AI 기반 신약 후보 물질 발굴',
    paperCount: 41,
    axes: ['데이터 편향', '약물 상호작용', '임상 전환'],
    gaps: [
      {
        title: '임상 전 데이터 편향 보정 부족',
        detail: 'in-vitro/in-vivo 데이터의 체계적 편향을 보정하여 AI 예측 정확도를 높이는 연구가 부족함',
        relatedPapers: 6,
      },
      {
        title: '다중 타겟 약물 상호작용 예측 한계',
        detail: '2개 이상의 타겟에 동시 작용하는 약물의 상호작용을 예측하는 모델의 성능과 해석성이 부족함',
        relatedPapers: 9,
      },
    ],
  },
]
