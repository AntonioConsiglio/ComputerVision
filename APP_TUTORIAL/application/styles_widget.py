
class ButtonStyle():

    def __init__(self):
        self.active_button = '''QPushButton
                                {
                                    background-color:#32CD32;
                                    border-style: outset;
                                    border-width: 1px;
                                    border-color: green;
                                    border-radius:15px;                                
                                }
                                QPushButton:hover
                                {
                                    background-color:#90EE90;
                                }
                                '''
        self.red_button = '''QPushButton
                                {
                                    background-color:red;
                                    border-style: outset;
                                    border-width: 1px;
                                    border-color: red;
                                    border-radius:15px;                                
                                }
                                QPushButton:hover
                                {
                                    background-color:yellow;
                                }
                                '''
        self.reset_button_active = '''QPushButton
                                    {
                                    background-color:#CD7F32;
                                    border-style: outset;
                                    border-width: 1px;
                                    border-color: 1E90FF;
                                    border-radius: 15px;							                                
                                    }
                                    QPushButton:hover
                                    {
                                    background-color:#D2B48C;
                                    }
                                    '''
        self.no_active_button = '''
                                '''
        

class LabelStyle():

    def __init__(self):
        self.red_string = '''QLabel
                                {
                                    color: red;
                                    border-width: 2px;
                                    border-color: red;                                
                                }
                                '''
