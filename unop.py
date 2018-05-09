# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:13:06 2018
@author: qli1
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup

class HTMLTableParser:
    def parse_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        return [(table,\
                #(table['id'],
                 self.parse_html_table(table))\
                for table in soup.find_all('table')]  

    def parse_html_table(self, table):
        n_columns = 0
        n_rows=0
        column_names = []

        # Find number of rows and columns
        # we also find the column titles if we can
        if table['id']!="opt_unusal_volume":
            pass
        else:
            for row in table.find_all('tr'):
               
                # Determine the number of rows in the table
                td_tags = row.find_all('td')
                if len(td_tags) > 0:
                    n_rows+=1
                    if n_columns == 0:
                        # Set the number of columns for our table
                        n_columns = len(td_tags)
                print(len(td_tags))    
                # Handle column names if we find them
                th_tags = row.find_all('th') 
                if len(th_tags) > 0 and len(column_names) == 0:
                    for th in th_tags:
                        if th.get_text()=='Option Volume Spikes':
                            pass
                        else:
     #                       print(th.get_text())
                            column_names.append(th.get_text())
    
            # Safeguard on Column Titles
    #        if len(column_names) > 0 and len(column_names) != n_columns:
    #             print(n_columns, len(column_names))           
    #             raise Exception("Column titles do not match the number of columns")
    
            columns = column_names if len(column_names) > 0 else range(0,n_columns)
            df = pd.DataFrame(columns = columns,
                              index= range(0,n_rows))
            row_marker = 0
            for row in table.find_all('tr'):
                column_marker = 0
                columns = row.find_all('td')
                for column in columns:
                    df.iat[row_marker,column_marker] = column.get_text()
                    column_marker += 1
                if len(columns) > 0:
                    row_marker += 1
                    
            # Convert to float if possible
            for col in df:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
            
        return df
    
hp = HTMLTableParser()
url = "https://www.fantasypros.com/nfl/reports/leaders/qb.php?year=2015"
url="https://marketchameleon.com/Reports/UnusualOptionVolumeReport"
tables= hp.parse_url(url)
table=tables[0][1]
        
print(table.head())
#try:
#    response = requests.get(url)
##    response.rasie_for_status()
#    print("done")
#    print(response.text[:100]) # Access the HTML with the text property
#except:
#    print("error")
#    
# while element of class "paginet_button_next" present" then click "next" button
#concat tbl_df from each page"





#html_string = '''
#      <table>
#            <tr>
#                <td> Hello! </td>
#                <td> Table </td>
#            </tr>
#        </table>
#    '''
#    
#soup = BeautifulSoup(html_string, 'lxml') # Parse the HTML as a string
#
#table = soup.find_all('table')[0] # Grab the first table
#
#new_table = pd.DataFrame(columns=range(0,2), index = [0]) # I know the size 
#
#row_marker = 0
#for row in table.find_all('tr'):
#    column_marker = 0
#    columns = row.find_all('td')
#    for column in columns:
#        new_table.iat[row_marker,column_marker] = column.get_text()
#        column_marker += 1
#
#print(new_table)
#https://www.youtube.com/watch?v=m_agcM_ds1c
#http://srome.github.io/Parsing-HTML-Tables-in-Python-with-BeautifulSoup-and-pandas/
