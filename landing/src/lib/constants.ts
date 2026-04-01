import {
  FileSearch, Brain, Target, Search, Database, Sparkles, FileText,
  Layers, Zap, BarChart3, Globe, Clock,
  type LucideIcon,
} from 'lucide-react'

// ── 네비게이션 ──
export const NAV_LINKS = [
  { label: '하는 일', href: '#service' },
  { label: '사용법', href: '#workflow' },
  { label: '핵심 기능', href: '#features' },
  { label: '차별점', href: '#difference' },
  { label: '예시', href: '#examples' },
] as const

// ── ServiceSection 카드 데이터 ──
export type ServiceCard = { icon: LucideIcon; title: string; description: string }

export const SERVICE_CARDS: ServiceCard[] = [
  {
    icon: FileSearch,
    title: '논문 수집 & 분석',
    description: 'arXiv, Semantic Scholar, OpenAlex, Crossref, ScienceON 등 6개 이상의 글로벌 학술 DB에서 키워드 기반으로 관련 논문을 자동 수집합니다. BM25 + FAISS 하이브리드 랭킹과 CrossEncoder 리랭킹으로 가장 관련성 높은 논문을 선별합니다.',
  },
  {
    icon: Brain,
    title: 'AI 기반 Research GAP 도출',
    description: '수집된 논문들의 한계점(Limitations)을 2-Track으로 추출(저자 명시 + 구조적 분석)한 뒤, AI가 연구 축(Research Axes)을 동적으로 생성하고 축별 장벽 분석을 통해 실질적인 연구 공백을 식별합니다.',
  },
  {
    icon: Target,
    title: '연구 방향 제안',
    description: '도출된 GAP을 기반으로 새로운 연구 주제와 구체적인 연구 방향을 제안합니다. Critic Agent가 결과를 평가하고, 품질이 부족하면 자동으로 재분석하여 신뢰도 높은 결과를 보장합니다.',
  },
]

// ── WorkflowSection 스텝 데이터 ──
export type WorkflowStep = { step: string; icon: LucideIcon; title: string; description: string }

export const WORKFLOW_STEPS: WorkflowStep[] = [
  { step: '01', icon: Search, title: '키워드 입력', description: '연구하고 싶은 주제나 관심 분야의 키워드를 입력하세요. 예: "LLM hallucination", "자율주행 안전성"' },
  { step: '02', icon: Database, title: '논문 자동 수집', description: '입력한 키워드를 기반으로 6개 학술 DB에서 관련 논문을 자동으로 검색하고 랭킹합니다.' },
  { step: '03', icon: Sparkles, title: 'GAP 분석 수행', description: 'AI가 수집된 논문들의 한계점을 추출하고, 연구 축별로 분류하여 비어 있는 연구 영역을 도출합니다.' },
  { step: '04', icon: FileText, title: '결과 리포트 확인', description: '도출된 Research Gap과 추천 연구 방향이 담긴 구조화된 리포트를 확인하고 추가 질문할 수 있습니다.' },
]

// ── FeaturesSection 기능 데이터 ──
export type FeatureItem = { icon: LucideIcon; title: string; description: string }

export const FEATURES: FeatureItem[] = [
  { icon: Globe, title: '글로벌 논문 DB 연동', description: 'arXiv, Semantic Scholar, OpenAlex, Crossref, ScienceON 등 6개 이상의 학술 데이터베이스에서 실시간으로 논문을 검색합니다. 영문/한글 논문 모두 지원합니다.' },
  { icon: Layers, title: '다중 논문 교차 분석', description: '수십 편의 논문을 동시에 분석하여 개별 논문에서는 보이지 않는 교차 패턴과 공통 한계점을 도출합니다. BM25 + FAISS 하이브리드 랭킹으로 관련성을 보장합니다.' },
  { icon: Zap, title: '실시간 AI 분석', description: '키워드 입력 후 수 분 내에 논문 수집부터 GAP 도출까지 전 과정을 자동 수행합니다. Multi-Agent 파이프라인이 병렬로 작업을 처리합니다.' },
  { icon: BarChart3, title: '구조화된 GAP 리포트', description: 'AI가 동적으로 생성한 연구 축(Research Axes)별로 GAP을 분류하고, 각 GAP의 근거 논문과 함께 구조화된 리포트를 제공합니다.' },
  { icon: Clock, title: '분석 이력 & 대화형 탐색', description: '이전 분석 결과를 저장하고 언제든 재확인할 수 있습니다. 분석 완료 후 GAP에 대해 추가 질문하며 심층 탐색이 가능합니다.' },
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
    feature: '논문 GAP 도출',
    gapago: true,
    gapagoDetail: 'Multi-Agent 파이프라인이 한계점 추출 → 연구 축 생성 → 장벽 분석을 자동 수행',
    existing: false,
    existingDetail: 'GAP 도출 기능 없음 (요약만 제공)',
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
    gapagoDetail: 'GAP 기반 구체적 연구 주제 및 방법론 제안 + Critic 검증',
    existing: false,
    existingDetail: '연구 방향 제안 기능 없음',
  },
  {
    feature: '구조화된 리포트',
    gapago: true,
    gapagoDetail: '연구 축별 GAP 분류 + 근거 논문 + 연구 방향이 포함된 체계적 리포트',
    existing: false,
    existingDetail: '단순 텍스트 요약만 제공',
  },
  {
    feature: '논문 수집 자동화',
    gapago: true,
    gapagoDetail: '6개 학술 DB 병렬 검색 + BM25/FAISS/CrossEncoder 3단계 랭킹',
    existing: true,
    existingDetail: '키워드 기반 검색은 가능하나 랭킹 정밀도 낮음',
  },
  {
    feature: '대화형 후속 탐색',
    gapago: true,
    gapagoDetail: '분석 결과에 대해 자연어로 추가 질문 및 심층 탐색 가능',
    existing: false,
    existingDetail: '정적 결과만 제공',
  },
]

// ── ExampleSection 예시 데이터 ──
export type ExampleCard = { query: string; gaps: string[] }

export const EXAMPLES: ExampleCard[] = [
  {
    query: '자율주행 안전성',
    gaps: [
      '센서 퓨전 환경에서의 엣지 케이스 처리 연구 부족',
      '악천후 환경 시뮬레이션 기반 검증 미비',
    ],
  },
  {
    query: 'LLM 할루시네이션',
    gaps: [
      '도메인 특화 할루시네이션 탐지 메트릭 부재',
      '멀티모달 환경에서의 사실 검증 방법론 미흡',
    ],
  },
  {
    query: '신약 개발 AI',
    gaps: [
      '임상 전 단계 데이터 편향 보정 연구 부족',
      '다중 타겟 약물 상호작용 예측 모델 한계',
    ],
  },
]
