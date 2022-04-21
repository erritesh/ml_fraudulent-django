import pydnsbl

#To check IP reputation api_void 
def pydns_api():
    IP= '68.128.212.240'
    ip_checker = pydnsbl.DNSBLIpChecker()
    ip_reputation= ip_checker.check(IP)
    print(ip_reputation)

   




pydns_api()


