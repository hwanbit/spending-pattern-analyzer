import React, { useState, useEffect } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend } from 'recharts';
import { Plus, Trash2, Search, Sliders, Activity, User, PieChart, X } from 'lucide-react';

function App() {
    // Toast 상태 추가
    const [toast, setToast] = useState(null);

    // 기존 상태 관리
    const [categories, setCategories] = useState({});
    const [items, setItems] = useState([]);
    const [selectedMajor, setSelectedMajor] = useState('');
    const [selectedMiddle, setSelectedMiddle] = useState('');
    const [amount, setAmount] = useState('');
    const [kValue, setKValue] = useState(5);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Toast 표시 함수
    const showToast = (message, type = 'info') => {
        setToast({ message, type });
        setTimeout(() => setToast(null), 3000);
    };

    // 초기 데이터 로드
    useEffect(() => {
        fetch('http://localhost:5000/categories')
            .then(res => res.json())
            .then(data => {
                setCategories(data);
                const firstMajor = Object.keys(data)[0];
                if (firstMajor) setSelectedMajor(firstMajor);
            })
            .catch(err => {
                console.error("카테고리 로드 실패:", err);
                showToast("카테고리 로드에 실패했습니다.", "error");
            });
    }, []);

    // 아이템 추가
    const handleAddItem = () => {
        if (!selectedMajor || !amount) {
            showToast("카테고리와 금액을 입력해주세요.", "warning");
            return;
        }

        const newItem = {
            id: Date.now(),
            major: selectedMajor,
            middle: selectedMiddle || '기타',
            amount: parseInt(amount)
        };

        setItems([...items, newItem]);
        setAmount('');
        showToast("소비 내역이 추가되었습니다.", "success");
    };

    const handleRemoveItem = (id) => {
        setItems(items.filter(item => item.id !== id));
        showToast("내역이 삭제되었습니다.", "info");
    };

    // 분석 요청
    const fetchAnalysis = async () => {
        if (items.length === 0) {
            showToast("소비 내역을 하나 이상 추가해주세요!", "warning");
            return;
        }

        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    items: items,
                    k: kValue
                })
            });
            const data = await response.json();
            setResult(data);
            showToast("분석이 완료되었습니다!", "success");
        } catch (error) {
            console.error(error);
            showToast("분석 중 오류가 발생했습니다.", "error");
        } finally {
            setLoading(false);
        }
    };

    const handleMajorChange = (e) => {
        setSelectedMajor(e.target.value);
        setSelectedMiddle('');
    };

    // 차트 데이터 변환
    const getChartData = () => {
        if (!result || !result.debug_vector) return [];
        // 모든 중분류 리스트를 백엔드와 동일한 순서(가나다순)로 생성
        const allMiddleCategories = Object.values(categories).flat().sort();

        // 대분류별 합산 데이터를 담을 객체 초기화
        const aggregatedData = {};
        Object.keys(categories).forEach(major => {
            aggregatedData[major] = { A: 0, B: 0 };
        });

        // 벡터를 순회하며 대분류별로 값 더하기
        const userVec = result.debug_vector[0]; // 나의 벡터
        const groupVec = result.group_vector;   // 그룹 평균 벡터

        userVec.forEach((val, idx) => {
            const middleName = allMiddleCategories[idx]; // 해당 인덱스의 중분류 이름 찾기

            // 이 중분류가 속한 대분류 찾기 (categories state 활용)
            const majorName = Object.keys(categories).find(key =>
                categories[key].includes(middleName)
            );

            if (majorName && aggregatedData[majorName]) {
                aggregatedData[majorName].A += val; // 나의 값 누적
                aggregatedData[majorName].B += groupVec[idx]; // 그룹 값 누적
            }
        });

        // Recharts용 배열로 변환
        return Object.keys(aggregatedData).map(major => ({
            subject: major,
            A: (aggregatedData[major].A * 100).toFixed(1), // 퍼센트로 변환
            B: (aggregatedData[major].B * 100).toFixed(1),
            fullMark: 100
        }));
    };

    return (
        <div className="min-h-screen bg-white text-gray-900 flex flex-col font-square">
            {/* Toast 알림 */}
            {toast && (
                <div className={`fixed top-6 right-6 z-50 px-6 py-4 rounded-lg shadow-2xl flex items-center gap-3 animate-slide-down ${
                    toast.type === 'success' ? 'border-2 border-solid border-blue-600 bg-white text-blue-600' :
                        toast.type === 'error' ? 'border-2 border-solid border-red-600 bg-white text-red-600' :
                            toast.type === 'warning' ? 'border-2 border-solid border-amber-500 bg-white text-amber-500' :
                                'bg-gray-900 text-white'
                }`}>
                    <span className="font-medium">{toast.message}</span>
                    <button onClick={() => setToast(null)} className="ml-2">
                        <X size={18} />
                    </button>
                </div>
            )}

            {/* 헤더 */}
            <header className="border-b border-gray-200 bg-white">
                <div className="max-w-7xl mx-auto px-6 py-8">
                    <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
                        {/* 왼쪽: 타이틀 */}
                        <div>
                            <h1 className="text-4xl font-square tracking-tight">
                                SPENDING PATTERN
                                <br />
                                <span className="text-blue-600">ANALYZER</span>
                            </h1>
                            <p className="mt-3 text-gray-600 text-sm">AI 기반 소비 패턴 분석 시스템</p>
                        </div>

                        {/* 오른쪽: 서비스 소개 */}
                        <div className="max-w-[500px] space-y-2 text-[12px] text-gray-600 leading-relaxed">
                            <p>* 평소 월별 지출 내역을 입력하면 AI가 빅데이터 분석을 통해 당신과 소비 패턴이 가장 닮은 인구 그룹을 찾아 드립니다.</p>
                            <p>* 내 소비 성향을 완벽하게 묘사하는 재치 있는 페르소나 별명은 물론, 남들과 비교했을 때 나의 독특한 지출 습관과 절약 포인트까지 콕 집어 알려줍니다.</p>
                            <p>* 분석의 정밀도를 직접 조절해 가며, 내가 몰랐던 나의 진짜 소비 정체성을 재미있게 탐색해 보세요.</p>
                        </div>
                    </div>
                </div>
            </header>

            {/* 메인 컨텐츠 */}
            <main className="flex-1 max-w-7xl mx-auto px-6 py-12 w-full">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                    {/* 입력 패널 */}
                    <div className="lg:col-span-1 space-y-6">
                        {/* 소비 입력 카드 */}
                        <div className="border border-gray-200 rounded-sm p-6">
                            <div className="flex items-center gap-3 mb-6">
                                <div className="w-8 h-8 bg-black text-white flex items-center justify-center text-sm font-bold">
                                    01
                                </div>
                                <h2 className="text-lg font-bold tracking-tight">소비 내역 입력</h2>
                            </div>

                            <div className="space-y-4">
                                {/* 카테고리 선택 */}
                                <div>
                                    <label className="block text-xs font-bold text-gray-500 mb-2 uppercase tracking-wide">Category</label>
                                    <div className="grid grid-cols-2 gap-3">
                                        <select
                                            className="w-full px-4 py-3 border border-gray-300 bg-white text-sm focus:outline-none focus:border-black transition-colors"
                                            value={selectedMajor}
                                            onChange={handleMajorChange}
                                        >
                                            {Object.keys(categories).map(cat => (
                                                <option key={cat} value={cat}>{cat}</option>
                                            ))}
                                        </select>
                                        <select
                                            className="w-full px-4 py-3 border border-gray-300 bg-white text-sm focus:outline-none focus:border-black transition-colors"
                                            value={selectedMiddle}
                                            onChange={(e) => setSelectedMiddle(e.target.value)}
                                            disabled={!selectedMajor}
                                        >
                                            <option value="">(상세 선택)</option>
                                            {categories[selectedMajor]?.map(mid => (
                                                <option key={mid} value={mid}>{mid}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                {/* 금액 입력 */}
                                <div>
                                    <label className="block text-xs font-bold text-gray-500 mb-2 uppercase tracking-wide">Amount</label>
                                    <div className="relative">
                                        <input
                                            type="number"
                                            className="w-full px-4 py-3 border border-gray-300 text-sm focus:outline-none focus:border-black transition-colors"
                                            placeholder="0"
                                            value={amount}
                                            onChange={(e) => setAmount(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleAddItem()}
                                        />
                                        <span className="absolute right-4 top-3 text-gray-400 text-sm">KRW</span>
                                    </div>
                                </div>

                                <button
                                    onClick={handleAddItem}
                                    className="w-full bg-black text-white py-3 text-sm font-bold uppercase tracking-wider hover:bg-gray-800 transition-colors flex items-center justify-center gap-2"
                                >
                                    <Plus size={16} /> Add to List
                                </button>
                            </div>
                        </div>

                        {/* 리스트 및 설정 카드 */}
                        <div className="border border-gray-200 rounded-sm p-6">
                            <div className="flex items-center gap-3 mb-6">
                                <div className="w-8 h-8 bg-black text-white flex items-center justify-center text-sm font-bold">
                                    02
                                </div>
                                <h2 className="text-lg font-bold tracking-tight">내역 및 설정</h2>
                            </div>

                            {/* 리스트 헤더 */}
                            <div className="flex justify-between items-center mb-4 pb-3 border-b border-gray-200">
                                <span className="text-xs font-bold text-gray-500 uppercase tracking-wide">
                                    Items ({items.length})
                                </span>
                                <span className="text-sm font-bold text-blue-600">
                                    {items.reduce((acc, cur) => acc + cur.amount, 0).toLocaleString()} KRW
                                </span>
                            </div>

                            {/* 리스트 영역 */}
                            <div className="space-y-2 mb-6 max-h-[240px] overflow-y-auto">
                                {items.length === 0 ? (
                                    <div className="py-12 text-center text-gray-400 text-sm">
                                        내역을 추가해주세요
                                    </div>
                                ) : (
                                    items.map((item) => (
                                        <div key={item.id} className="flex justify-between items-center border border-gray-200 p-3 hover:border-black transition-colors">
                                            <div className="flex-1">
                                                <div className="font-bold text-sm">{item.major}</div>
                                                <div className="text-xs text-gray-500">{item.middle}</div>
                                            </div>
                                            <div className="flex items-center gap-4">
                                                <span className="font-mono text-sm">{item.amount.toLocaleString()}</span>
                                                <button
                                                    onClick={() => handleRemoveItem(item.id)}
                                                    className="text-gray-400 hover:text-red-600 transition-colors"
                                                >
                                                    <Trash2 size={16} />
                                                </button>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>

                            {/* 분석 설정 */}
                            {items.length > 0 && (
                                <div className="space-y-4 pt-4 border-t border-gray-200">
                                    <div>
                                        <div className="flex justify-between items-center mb-3">
                                            <span className="text-xs font-bold text-gray-500 uppercase tracking-wide">
                                                Analysis Precision
                                            </span>
                                            <span className="text-xs font-bold text-blue-600 bg-blue-50 px-2 py-1">
                                                K = {kValue}
                                            </span>
                                        </div>
                                        <input
                                            type="range" min="3" max="8" step="1"
                                            value={kValue}
                                            onChange={(e) => setKValue(parseInt(e.target.value))}
                                            className="w-full h-1 bg-gray-200 appearance-none cursor-pointer accent-black"
                                        />
                                        <div className="flex justify-between text-[10px] text-gray-400 mt-2 uppercase tracking-wide">
                                            <span>Broad</span>
                                            <span>Precise</span>
                                        </div>
                                    </div>

                                    <button
                                        onClick={fetchAnalysis}
                                        disabled={loading}
                                        className="w-full bg-black text-white py-4 text-sm font-bold uppercase tracking-wider hover:bg-gray-800 transition-colors flex items-center justify-center gap-2 disabled:bg-gray-400"
                                    >
                                        {loading ? (
                                            <span className="animate-pulse">Analyzing...</span>
                                        ) : (
                                            <>
                                                <Search size={16} /> Start Analysis
                                            </>
                                        )}
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* 결과 패널 */}
                    <div className="lg:col-span-2">
                        {!result ? (
                            <div className="h-full min-h-[600px] border-2 border-dashed border-gray-300 flex flex-col items-center justify-center text-center p-12">
                                <Activity size={64} className="mb-6 text-gray-300" />
                                <h3 className="text-xl font-bold text-gray-400 mb-3 uppercase tracking-tight">
                                    Awaiting Analysis
                                </h3>
                                <p className="text-sm text-gray-400 max-w-[500px]">
                                    왼쪽 패널에서 소비 내역을 입력하고 분석 정밀도를 설정한 후 분석을 시작해주세요.
                                </p>
                            </div>
                        ) : (
                            <div className="space-y-6">
                                {/* 페르소나 결과 - 01번 박스와 높이 맞춤 */}
                                <div className="border-4 border-black p-8 relative overflow-hidden h-[317px] flex flex-col">
                                    <div className="absolute top-4 right-4 text-xs font-bold text-gray-400 uppercase tracking-wider">
                                        Analysis Level {kValue}
                                    </div>

                                    <div className="mb-auto">
                                        <div className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-2">
                                            Your Spending Persona
                                        </div>
                                        <h2 className="text-3xl md:text-4xl font-bold leading-tight mb-7">
                                            {result.persona_nickname}
                                        </h2>
                                        <div className="text-sm">
                                            {result.persona_tags}
                                        </div>
                                    </div>

                                    {/* Gap Analysis */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-auto">
                                        <div className="border border-gray-300 p-4">
                                            <div className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">
                                                🔥 Unique Trait
                                            </div>
                                            <p className="text-sm font-medium">{result.gap_analysis?.unique_trait}</p>
                                        </div>
                                        <div className="border border-gray-300 p-4">
                                            <div className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">
                                                💰 Saving Point
                                            </div>
                                            <p className="text-sm font-medium">{result.gap_analysis?.saving_trait}</p>
                                        </div>
                                    </div>
                                </div>

                                {/* 차트 영역 */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* 레이더 차트 */}
                                    <div className="border border-gray-200 p-6">
                                        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-6 flex items-center gap-2">
                                            <PieChart size={14} /> Pattern Comparison
                                        </h3>
                                        <div className="h-64">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={getChartData()}>
                                                    <PolarGrid stroke="#e5e7eb" />
                                                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#6b7280', fontSize: 10 }} />
                                                    <PolarRadiusAxis angle={30} domain={[0, 50]} tick={false} axisLine={false} />
                                                    <Radar name="나" dataKey="A" stroke="#000000" strokeWidth={2} fill="#3b82f6" fillOpacity={0.3} />
                                                    <Radar name="그룹 평균" dataKey="B" stroke="#9ca3af" strokeWidth={2} strokeDasharray="4 4" fill="transparent" />
                                                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                                                </RadarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>

                                    {/* 추가 정보 */}
                                    <div className="border border-gray-200 p-6 flex flex-col justify-center items-center text-center">
                                        <h4 className="text-sm font-bold mb-3 uppercase tracking-tight">
                                            Adjust Precision
                                        </h4>
                                        <p className="text-xs text-gray-600 mb-6 leading-relaxed">
                                            현재 K={kValue} 결과가 만족스러우신가요?<br/>
                                            왼쪽 패널에서 슬라이더를 조절하고 다시 분석해보세요.<br/>
                                            섬세한 조절로 다른 결과가 도출됩니다.
                                        </p>
                                        <div className="inline-block bg-gray-100 px-4 py-2 text-xs text-gray-500 uppercase tracking-wide">
                                            Re-analyze for Different Results
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 bg-white mt-auto">
                <div className="max-w-7xl mx-auto px-6 py-6">
                    <p className="text-center text-sm text-gray-500">
                        © 2025 Elphie. All rights reserved.
                    </p>
                </div>
            </footer>

            <style>{`
                @keyframes slide-down {
                    from {
                        transform: translateY(-100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                .animate-slide-down {
                    animation: slide-down 0.3s ease-out;
                }
            `}</style>
        </div>
    );
}

export default App;