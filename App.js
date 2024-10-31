import styled from 'styled-components/native';
import {Text, View, Image, SafeAreaView} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import TabNavigator from './navigation/navi';
import Dashboard from './Dashboard';
import MyLayout from './layout';

export default function App() {
  return (
    <View style={{backgroundColor: '#14161c', flex:1,}}>
      <MyLayout/>
    </View>
  );
}
