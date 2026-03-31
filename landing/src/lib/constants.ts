import {
  FileSearch, Brain, Target, Search, Database, Sparkles, FileText,
  Layers, Zap, BarChart3, Globe, Lock, Clock,
  type LucideIcon,
} from 'lucide-react'

// ── 네비게이션 ──
export const NAV_LINKS = [
  { label: '기능', href: '#features' },
  { label: '사용법', href: '#workflow' },
  { label: '차별점', href: '#difference' },
] as const

// ── ServiceSection 카드 데이터 ──
export type ServiceCard = { icon: LucideIcon; title: string; description: string }

export const SERVICE_CARDS: ServiceCard[] = [
  {
    icon: FileSearch,
    title: '논문 수집 & 분석',
    description: '키워드 기반으로 관련 논문을 자동 수집하고 핵심 내용을 구조화합니다.',
  },
  {
    icon: Brain,
    title: 'AI기반 GAP 도출',
    description: '단순 요약이 아닌, 연구 흐름 속 비어 있는 영역을 AI가 자동으로 식별합니다.',
  },
  {
    icon: Target,
    title: '연구 방향 제안',
    description: '도출된 GAP을 기반으로 새로운 연구 주제와 방향을 제안합니다.',
  },
]

// ── WorkflowSection 스텝 데이터 ──
export type WorkflowStep = { step: string; icon: LucideIcon; title: string; description: string }

export const WORKFLOW_STEPS: WorkflowStep[] = [
  { step: '01', icon: Search, title: '키워드 입력', description: '연구 주제나 관심 분야를 입력합니다' },
  { step: '02', icon: Database, title: '논문 수집', description: '관련 논문을 자동으로 검색·수집합니다' },
  { step: '03', icon: Sparkles, title: 'GAP 분석', description: 'AI가 연구 공백을 자동 도출합니다' },
  { step: '04', icon: FileText, title: '결과 확인', description: '구조화된 GAP 리포트를 제공합니다' },
]

// ── FeaturesSection 기능 데이터 ──
export type FeatureItem = { icon: LucideIcon; title: string; description: string }

export const FEATURES: FeatureItem[] = [
  { icon: Layers, title: '다중 논문 동시 분석', description: '여러 논문을 한 번에 분석하여 교차 검증된 GAP을 도출합니다.' },
  { icon: Zap, title: '실시간 처리', description: '키워드 입력 후 수 분 내에 분석 결과를 확인할 수 있습니다.' },
  { icon: BarChart3, title: '시각적 리포트', description: 'GAP 분석 결과를 직관적인 차트와 구조로 시각화합니다.' },
  { icon: Globe, title: '글로벌 논문 DB', description: '주요 학술 데이터베이스에서 최신 논문을 검색합니다.' },
  { icon: Lock, title: '데이터 보안', description: '분석 데이터는 안전하게 암호화되어 보호됩니다.' },
  { icon: Clock, title: '분석 이력 관리', description: '이전 분석 결과를 저장하고 언제든 다시 확인합니다.' },
]

// ── DifferenceSection 비교 데이터 ──
export type ComparisonRow = { feature: string; gapago: boolean; existing: boolean }

export const COMPARISON_DATA: ComparisonRow[] = [
  { feature: '논문 GAP 도출', gapago: true, existing: false },
  { feature: '단순 논문 요약', gapago: true, existing: true },
  { feature: '다중 논문 교차 분석', gapago: true, existing: false },
  { feature: '연구 방향 제안', gapago: true, existing: false },
  { feature: '시각적 리포트', gapago: true, existing: false },
  { feature: '키워드 기반 자동 수집', gapago: true, existing: true },
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
