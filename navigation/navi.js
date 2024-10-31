import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import StatusScreen from '../components/StatusScreen';
import ToolsScreen from '../components/ToolsScreen';
import AntDesign from '@expo/vector-icons/AntDesign';
import Entypo from '@expo/vector-icons/Entypo';

const Tab = createBottomTabNavigator();

export default function TabNavigator() {
    return (
        <Tab.Navigator
            initialRouteName='Status'
            screenOptions={{
                tabBarActiveTintColor: '#e91e63',
            }}>
            <Tab.Screen name="Status" component={StatusScreen} options={{headerShown: false,
                tabBarIcon: () => (
                    <AntDesign name="dashboard" size={24} color="black" />
                ),
            }}/>
            <Tab.Screen name="Tools" component={ToolsScreen} options={{headerShown: false,
                tabBarIcon: () => (
                    <Entypo name="tools" size={24} color="black" />
                ),
            }}/>
        </Tab.Navigator>
    );
}