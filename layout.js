import React, {useState, useEffect} from 'react';
import {View, ScrollView, Button, Text, TextInput, Image, StyleSheet, Dimensions, useWindowDimensions } from 'react-native';
import {SpeedoMeterPlus} from 'react-native-speedometer-plus';
const {initializeApp} = require("firebase/app");
const {getDatabase, ref, set, onValue} = require("firebase/database");
import { LineChart, BarChart  } from 'react-native-chart-kit';
import Speedometer, {Background, Arc, Needle, Progress, Marks, Indicator} from 'react-native-cool-speedometer';


export default function MyLayout(){

    const firebaseConfig = {
        apiKey: "AIzaSyBYc1K57qqyIBHp1K7ZlKnZvJDpZzEzK4A",
        authDomain: "gpudashboa.firebaseapp.com",
        databaseURL: "https://gpudashboa-default-rtdb.asia-southeast1.firebasedatabase.app",
        projectId: "gpudashboa",
        storageBucket: "gpudashboa.appspot.com",
        messagingSenderId: "128148028990",
        appId: "1:128148028990:web:805bdd8cc255bf46e6db6f",
        measurementId: "G-ZXMQG8Z2P3"
      };

    const firebase = initializeApp(firebaseConfig);
    const database = getDatabase(firebase);

    const ClockLabels = [
        {
            name: 'Test0',
            labelColor: '#3b699e',
            activeBarColor: '#93c2ee',
          },
        {
            name: 'Test1',
            labelColor: '#3b699e',
            activeBarColor: '#7dacda',
          },

          {
            name: 'Test2',
            labelColor: '#3b699e',
            activeBarColor: '#6696c6',
          },
          {
            name: 'Test3',
            labelColor: '#3b699e',
            activeBarColor: '#507fb2',
          },
          {
            name: 'Test4',
            labelColor: '#3b699e',
            activeBarColor: '#3b699e',
          },
          
          
          
    ]
    const TempLabels = [
        {
            name: 'Test0',
            labelColor: '#ff5153',
            activeBarColor: '#ffd9db',
          },
        {
            name: 'Test1',
            labelColor: '#ff5153',
            activeBarColor: '#ffb7b9',
          },

          {
            name: 'Test2',
            labelColor: '#ff5153',
            activeBarColor: '#ff9597',
          },
          {
            name: 'Test3',
            labelColor: '#ff5153',
            activeBarColor: '#ff7375',
          },
          {
            name: 'Test4',
            labelColor: '#ff5153',
            activeBarColor: '#ff5153',
          },
    ]
    const PowerLabels = [
        {
            name: 'Test0',
            labelColor: '#02a361',
            activeBarColor: '#cce3dc',
          },
        {
            name: 'Test1',
            labelColor: '#02a361',
            activeBarColor: '#99d3bd',
          },

          {
            name: 'Test2',
            labelColor: '#02a361',
            activeBarColor: '#66c39e',
          },
          {
            name: 'Test3',
            labelColor: '#02a361',
            activeBarColor: '#33b380',
          },
          {
            name: 'Test4',
            labelColor: '#02a361',
            activeBarColor: '#02a361',
          },
          
    ]

    const [inputValue, setInputValue] = useState('');

    const [clock, setClock] = useState(0);
    const [temp, setTemp] = useState(0);
    const [power, setPower] = useState(0);
    const [clockChartData, setClockChartData] = useState([0]);
    const [tempChartData, setTempChartData] = useState([0]);
    const [powerChartData, setPowerChartData] = useState([0]);

    const [clockChartLen, setClockChartLen] = useState(0);
    const [tempChartLen, setTempChartLen] = useState(0);
    const [powerChartLen, setPowerChartLen] = useState(0);

    const [defaultPower, setDefaultPower] = useState(0);
    const [OptiPower, setOptiPower] = useState(0);
    const [defaultPer, setDefaultPer] = useState(0);
    const [OptiPer, setOptiPer] = useState(0);

    const { width } = useWindowDimensions();
    const boxSize = width * 0.45; // 박스 너비의 45%를 비율로 설정
    const chartSize = width * 0.3; // 차트 너비의 80%를 비율로 설정
    const dynamicMargin = width * 0.05;

    const getBarColor = (index) => {
        const colors = ['#3b699e', '#02a361']; // 각 막대의 색상
        return colors[index] || '#000'; // 기본 색상
      };

    useEffect(()=> {
        const firstRef = ref(database, 'firstEpoch')
        const firstListener = onValue(firstRef, (snapshot) => {
            console.log("firstListener")
            const data = snapshot.val();
            const done = data.done
            if (done) {
                setDefaultPower(parseInt(data.powerDraw * data.epoch))
                setDefaultPer(parseInt(data.performance * data.epoch))
            }


                
        })

        const epochRef = ref(database, 'epochInfo')
        const epochListener = onValue(epochRef, (snapshot) => {
            console.log("epoch")
            const data = snapshot.val();
            setOptiPower(prev => prev + data.powerDraw)
            setOptiPer(prev => prev + data.performance)
            
        })

        const gpuLogsRef = ref(database, 'gpuLogs');

        const logListener = onValue(gpuLogsRef, (snapshot) => {
            const data = snapshot.val();
            const newClock = parseInt(data.clock, 10);
            const newTemp = parseInt(data.temperature, 10);
            const newPower = parseInt(data.powerConsumption, 10);
            setClock(newClock);
            setTemp(newTemp);
            setPower(newPower);

            setClockChartData(prevData => {
                const updatedData = [...prevData, newClock];
                setClockChartLen(prevLen => prevLen + 1); 
                
                return updatedData.length > 20 ? updatedData.slice(1) : updatedData;
                //return updatedData;
            });

            setTempChartData(prevData => {
                const updatedData = [...prevData, newTemp];
                setTempChartLen(prevLen => prevLen + 1);
                return updatedData.length > 20 ? updatedData.slice(1) : updatedData;
                //return updatedData;
            });

            setPowerChartData(prevData => {
                const updatedData = [...prevData, newPower];
                setPowerChartLen(prevLen => prevLen + 1);
                return updatedData.length > 20 ? updatedData.slice(1) : updatedData;
                //return updatedData;
            });


            //console.log(`clock : ${newClock}`);
        });

        return () => {
            logListener();
            firstListener();
            epochListener();
        };
    }, [])
   
  return (
    

        <View style={styles.container}>
            {/* 제목 구역 */}
            
            <Text style={styles.titleText}>GPU Dashboard</Text>
            
        {/* 상단 4개의 구역 */}
        <View style={styles.topContainer}>
            <View style={styles.box}>
                <SpeedoMeterPlus
                                value={clock}
                                size={boxSize * 0.233}
                                minValue={0}
                                maxValue={3000}
                                innerLabelNoteValue="CLOCK"
                                labels = {ClockLabels}
                                innerLabelNoteStyle={{ color: '#3b699e', fontSize: 23 }}
                                
                                
                    />
                

            </View>
            <View style={styles.box}>
            <SpeedoMeterPlus
                            value={temp}
                            size={boxSize * 0.233}
                            minValue={0}
                            maxValue={100}
                            innerLabelNoteValue="TEMP"
                            labels = {TempLabels}
                            innerLabelNoteStyle={{ color: '#ff5153', fontSize: 23 }}
                        />
            </View>
            <View style={styles.box}>
            <SpeedoMeterPlus
                            value={power}
                            size={boxSize * 0.233}
                            minValue={0}
                            maxValue={200}
                            innerLabelNoteValue="POWER"
                            labels = {PowerLabels}
                            innerLabelNoteStyle={{ color: '#02a361', fontSize: 23 }}
                        />
            </View>
            <View style={styles.infoBox}>
                <View style={{'marginLeft':80}}>

                <BarChart
                    data={{
                        labels: ['Default', 'Optimized'], // X축 레이블
                        datasets: [
                          {
                            data: [defaultPower, OptiPower], // Y축 값
                          },
                        ],
                      }}
                    width={chartSize * 0.4} // 화면 너비에서 여백을 뺀 값
                    height={chartSize * 0.4} // 차트 높이
                    yAxisLabel="" // Y축 레이블
                    chartConfig={{
                    backgroundColor: '#ffffff', // 배경색
                    backgroundGradientFrom: '#ffffff', // 그라데이션 시작색
                    backgroundGradientTo: '#ffffff', // 그라데이션 끝색
                    decimalPlaces: 2, // 소수점 자리수
                    color: (opacity = 0) => `rgba(102, 255, 51, ${opacity})`, // 바 색상
                    labelColor: (opacity = 1) => `rgba(59, 105, 158, ${opacity})`, // 레이블 색상
                    style: {
                        borderRadius: 16, // 테두리 반경
                    },
                    propsForDots: {
                        r: '6',
                        strokeWidth: '2',
                        stroke: '#ffa726',
                    },
                    }}
                    style={{
                    marginVertical: 7,
                    borderRadius: 16,
                    
                    }}
                    
                    fromZero={true} // 시작점을 0으로 설정
                />
                </View>





                {/* 수직으로 텍스트 배치 */}
                <View style={styles.textContainer}>
                    <View style={{flexDirection: 'cols', paddingVertical : 0}}>
                        <Text style={styles.infoText}>{defaultPower}W</Text>
                        <View style={{flexDirection: 'row', alignItems: 'center'}}>
                            <Text style={{fontSize: 20, 
                            fontWeight: '100',
                            color: '#3b699e', marginRight:10}}>Default            </Text>
                            <Text style={styles.infoTextPer}>{parseInt(defaultPer)}s</Text>
                        </View>
                    </View>
                    <View style={{flexDirection: 'cols', marginBottom:10}}>
                        <Text style={[styles.infoText, {color:'#02a361'}]}>{parseInt(OptiPower)}W</Text>
                        <View style={{flexDirection: 'row', alignItems: 'center'}}>

                        <Text style={{fontSize: 20, 
                        fontWeight: '100',
                        color: '#02a361', marginRight : 10}}>Optimized       </Text>
                        <Text style={[styles.infoTextPer, {color:'#02a361'}]}>{parseInt(OptiPer)}s</Text>
                        </View>
                    </View>
                    
                </View>
            </View>
        </View>
        
        {/* 하단 큰 구역 */}
        <View style={styles.bottomContainer}>
            <View style={styles.bottomBox}>
                <Text style={[styles.chartTitle, {color: 'black'}]}>Clock</Text>
                
           
                    
                        <LineChart
                            data={{
                                labels: clockChartLen > 20
                                ? Array.from({ length: 20 }, (_, index) => (clockChartLen - 20 + index + 1).toString())
                                : clockChartData.map((_, index) => (index + 1).toString()), // 기본 레이블 1부터 시작
                                datasets: [
                                    {
                                        data: clockChartData,
                                        strokeWidth: 4,
                                    },
                                ],
                            }}
                            width={chartSize} // react-native에서
                            height={chartSize * 0.7} 
                            yAxisLabel=""
                            yAxisSuffix=" Hz"
                            chartConfig={{
                                backgroundColor: '#ffffff',
                                backgroundGradientFrom: '#ffffff',
                                backgroundGradientTo: '#ffffff',
                                decimalPlaces: 0,
                                color: (opacity = 1) => `rgba(59, 105, 158, ${opacity})`,
                                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                                style: {
                                    borderRadius: 16,
                                    borderWidth: 0,
                                },
                                propsForDots: {
                                    r: 0, // 동그라미 표시 제거
                                    fill: 'none', // 색상 없애기
                                },
                                useShadowColorFromDataset: true,
                                fillShadowGradientFrom: '#ffffff', // 투명 음영
                                fillShadowGradientOpacity: 0, // 음영 투명도 0으로 설정
                            }}
                            fromZero={true}
                        />
          
                
            </View>
            <View style={styles.bottomBox}>
                <Text style={[styles.chartTitle, {color : 'black'}]}>Temp</Text>
               
                <LineChart
                        data={{
                            labels: tempChartLen > 20
                            ? Array.from({ length: 20 }, (_, index) => (tempChartLen - 20 + index + 1).toString())
                            : tempChartData.map((_, index) => (index + 1).toString()), // 기본 레이블 1부터 시작
                            datasets: [
                                {
                                    data: tempChartData,
                                    strokeWidth: 4,
                                },
                            ],
                        }}
                        width={chartSize} // react-native에서
                        height={chartSize*0.7}
                        yAxisLabel=""
                        yAxisSuffix="°C"
                        yAxisInterval={10} // y축 간격 설정 (예: 1)
                        chartConfig={{
                            backgroundColor: '#ffffff',
                            backgroundGradientFrom: '#ffffff',
                            backgroundGradientTo: '#ffffff',
                            decimalPlaces: 0,
                            color: (opacity = 1) => `rgba(255, 81, 83, ${opacity})`,
                            labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                            style: {
                                borderRadius: 16,
                            },
                            propsForDots: {
                                r: 0, // 동그라미 표시 제거
                                fill: 'none', // 색상 없애기
                            },
                            useShadowColorFromDataset: true,
                            fillShadowGradientFrom: '#ffffff', // 투명 음영
                            fillShadowGradientOpacity: 0, // 음영 투명도 0으로 설정
                        }}
                        fromZero={true}
                    />
               
                
            </View>
            
            <View style={styles.bottomBox}>
                <Text style={[styles.chartTitle, {color : 'black'}]}>Power</Text>
                
                    <LineChart
                            data={{
                                labels: powerChartLen > 20
                                ? Array.from({ length: 20 }, (_, index) => (powerChartLen - 20 + index + 1).toString())
                                : powerChartData.map((_, index) => (index + 1).toString()), // 기본 레이블 1부터 시작
                                datasets: [
                                    {
                                        data: powerChartData,
                                        strokeWidth: 4,
                                    },
                                ],
                            }}
                            width={chartSize} // react-native에서
                            height={chartSize*0.7}
                            yAxisLabel=""
                            yAxisSuffix="W"
                            chartConfig={{
                                backgroundColor: '#ffffff',
                                backgroundGradientFrom: '#ffffff',
                                backgroundGradientTo: '#ffffff',
                                decimalPlaces: 0,
                                color: (opacity = 1) => `rgba(2, 163, 97, ${opacity})`,
                                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                                style: {
                                    borderRadius: 16,
                                },
                                propsForDots: {
                                    r: 0, // 동그라미 표시 제거
                                    fill: 'none', // 색상 없애기
                                },
                                useShadowColorFromDataset: true,
                                fillShadowGradientFrom: '#ffffff', // 투명 음영
                                fillShadowGradientOpacity: 0, // 음영 투명도 0으로 설정
                            }}
                            fromZero={true}
                        />
                
                
            </View>
                            
            


        </View>
        </View>
    
  );
};

const styles = StyleSheet.create({
    container: { flex: 1, padding: 20, backgroundColor: '#f0f5f9' },
    titleBox: { paddingVertical: 15, backgroundColor: '#fff', marginBottom: 15, borderRadius: 10, alignItems: 'flex-start',  // 위쪽 테두리 두께 설정
            marginHorizontal: 14,},
    titleText: { fontSize: 24, fontWeight: '200', color: '#3b699e', marginLeft : 30, marginBottom : 10 },
    topContainer: { flexDirection: 'row', justifyContent: 'space-between', margin : 10 },
    bottomContainer : {flexDirection: 'row', justifyContent: 'space-between', margin : 10, marginTop:70},
    box: { flex: 1, backgroundColor: '#fff', padding: 0, borderRadius: 10, alignItems: 'center', marginHorizontal: 7, paddingTop: 50, // 위쪽 테두리 두께 설정
        },
    
    logoBox:{flex : 0.8, backgroundColor: '#fff', padding: 15, borderRadius: 10, alignItems: 'center', marginHorizontal: 7, paddingTop: 50, // 위쪽 테두리 두께 설정
        },
    infoBox: { flex: 1.5, backgroundColor: '#fff', borderRadius: 10, padding: 0, alignItems: 'center', marginHorizontal: 5, flexDirection:'row',  // 위쪽 테두리 두께 설정
         },

    bottomBox: { flex: 1, backgroundColor: '#fff', padding: 0, borderRadius: 10, alignItems: 'center', marginHorizontal: 7, paddingTop: 0, // 위쪽 테두리 두께 설정
        },
    chartContainer: { marginRight: 20, alignItems: 'center' },
    chartTitle: { fontSize: 20, fontWeight: '500', color: '#333', marginLeft:40 , marginTop:10, marginBottom:10, textAlign: 'left',  // 텍스트 왼쪽 정렬
        width: '100%', },
    
    labelStyle: { fontSize: 18, color: '#333' },
    textContainer: {flexDirection:'col', alignItems:'flex-start', marginLeft:40,},
    infoText: { 
        fontSize: 40, 
        fontWeight: '400',
        color: '#3b699e',
        marginVertical: 4, // 각 텍스트 간의 여백
      },
    chartText : {fontSize: 40, 
    fontWeight: '100',
    color: '#3b699e',
    marginVertical: 4, textAlign:'left',  width: '100%',marginLeft:40 ,},
    infoTextPer : {
        fontSize : 15,
        fontWeight : 100,
        color: '#3b699e',
        
    }
});
