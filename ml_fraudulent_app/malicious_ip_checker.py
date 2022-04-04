#!/usr/local/python3

import pdb
import json
import socket
import requests
import os
from ipwhois import IPWhois
from colorama import Fore,Back,Style,init
init()

class MaliciousIPCheck():

        def __init__(self,url_ip):
                self.url_ip=url_ip
                ipaddr=socket.gethostbyname(url_ip)
                self.ipaddr=ipaddr
                print("IP of the URL: {}".format(ipaddr))
                print("---------------------------------------------------------------")

        def local_database(self):
                print("No. of times this IP previously appeared attempting against our organization:")
                os.system("cat /usr/local/src/prev_reported.txt | grep {} | wc -l".format(self.ipaddr))

        def whois_info(self): #Abuse Contact from WHOIs
                out=IPWhois(self.ipaddr)
                whoisthere=out.lookup_whois()
                final=json.dumps(whoisthere,indent=4)
                with open ("/var/tmp/whoisthere.json","w") as file:
                        file.write(final)
                print("---------Abuse Contact Information from Whois Database---------")
                print("Abuse Contact:")
                os.system("cat /var/tmp/whoisthere.json | jq -C '.nets[0].emails'")

        def virus_total(self): #VirusTotal
                apiurl="https://www.virustotal.com/api/v3/ip_addresses/"+self.ipaddr
                header={
                "Accept":"application/json",
                "x-apikey":"---This is just an example, please register for a valid API Key---"
                }
                r=requests.request(method='GET',url=apiurl,headers=header)
                decodedresponse=json.loads(r.text)
                resp=json.dumps(decodedresponse,sort_keys=True,indent=4)
                with open ("/var/tmp/virustotal.json","w")as file:
                        file.write(resp)

                print("--------------------Virus Total Information--------------------")
                print(Fore.RED+"Flagged as Malware by no. of engines:")
                os.system("cat /var/tmp/virustotal.json | jq '.data.attributes.last_analysis_results[].result' | grep malware | wc -l")
                print(Fore.RED+"No.of times flagged as malicious:")
                os.system("cat /var/tmp/virustotal.json | jq -C '.data.attributes.last_analysis_results[].result' | grep malicious | wc -l")

        def abuse_ipdb(self): #AbuseIPDB
                url='https://api.abuseipdb.com/api/v2/check'
                reporturl="https://api.abuseipdb.com/api/v2/report"
                qs={
                        'ipAddress':self.ipaddr,
                        'maxAgeInDays':'90'
                }
                report={
                        'ip':self.ipaddr,
                        'categories':'18,19',
                        #'comment':'Malicious Attempts'
                }
                header={
                        'Accept':'application/json',
                        'Key':'---This is just an example, please register for a valid API Key---'
                }
                init_response=requests.request(method='GET',url=url,headers=header,params=qs)
                decodedresponse=json.loads(init_response.text)
                res=json.dumps(decodedresponse,sort_keys=True,indent=4)
                with open('/var/tmp/abuseipdb.json','w') as file:
                        file.write(res)

                print(Fore.WHITE+"----------------Information from AbuseIPDB---------------------")
                print("Domain:")
                os.system("cat /var/tmp/abuseipdb.json | jq -C '.'data.domain''")
                print("Number of Times Reported to AbuseIPDB:")
                reported=os.system("cat /var/tmp/abuseipdb.json | jq -C '.'data.totalReports''")

                a=input("Do you want to report this IP to AbuseIPDB[y/Y or n/N]:")
                if a=='y' or a=='Y':
                        post_response=requests.request(method='POST',url=reporturl,headers=header,params=report)
                        py_conv=json.loads(post_response.text)
                        respon=json.dumps(py_conv,sort_keys=True,indent=4)
                        with open('/var/tmp/abipdb_resp.json','w')as myfile:
                                myfile.write(respon)
                        print("Abuse Confidence Score from AbuseIPDB:")
                        os.system("cat /var/tmp/abipdb_resp.json | jq -C '.'data.abuseConfidenceScore''")
                elif a=='n' or a=='N':
                        print("Not reported to AbuseIPDB")
        
        def blocking_ip(self):
            #Command to Block it on our own database can be added here
            yesorno=input("Do you want to block the above IP and mark it in our own database: (Please make sure it is not hosted on Google, Amazon or Microsoft)[y/Y or n/N]:")
            if yesorno=='y' or yesorno=='Y':
                with open("/usr/local/src/prev_reported.txt","a+") as myfile:
                    myfile.write(self.ipaddr+"\n")
            else:
                return

condn=True
while condn:
        try:
                info=input("Please enter the URL or IP to check:")
                mic=MaliciousIPCheck(info)
                mic.local_database()
                print(Fore.WHITE+"\n")
                mic.whois_info()
                mic.virus_total()
                mic.abuse_ipdb()
                mic.blocking_ip()
                print("\n")
                cont=input("Do you want to continue checking for another IP:[y/n]:")
                if cont=='y':
                        continue
                else:
                        condn=False
        except socket.gaierror:
                print("Not http/https format,rather just the domain name[e.g: google.com]")
        #else:
                #print("Encountered an error, please enter the domain or IP again or press Ctrl+z to exit!")