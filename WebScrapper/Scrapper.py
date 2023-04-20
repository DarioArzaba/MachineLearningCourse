#  pip install bs4 lxml
from bs4 import BeautifulSoup
import os, numpy, time, datetime

def pageProcess():

	os.system('curl "https://www.nissanseminuevos.com.mx/searchused.aspx?Type=U&Model=FRONTIER&pn=50" --output 0.html -s');

	with open('0.html', 'r') as f:
		contents = f.read();

	soup = BeautifulSoup(contents, 'html.parser');

	S = [];
	P = [];
	VIN = [];

	for tag in soup.find_all('li'):
		key = f"{tag.text}";
		if key.find('Precio:') != -1:
			P.append(key);
		#
		if key.find('Kilómetros:') != -1:
			S.append(key);		
		#
		if key.find('Número de VIN:') != -1:
			VIN.append(key);		
		#


	
	#S = S[1:];
	S = S[0:-1:2];
	
	P = P[1:];
	P = P[0:-1:2];
	
	VIN = VIN[0:-1:2];
	

	
	S = [int(x.split(': ')[-1].replace(',','')) for x in S ];
	
	Q = [];
	
	for x in P:
		try:
			x = x.split('Precio: ')[-1];
			x = x.replace(',','');
			x = x.replace('$','');
			x = int(x);
		except:
			x = -1;
		Q.append(x)
	
	P = Q;
	#P = [int(x.split(': $')[-1].replace(',','')) for x in P ];
	VIN = [x.split(': ')[-1].replace(',','') for x in VIN ];

	print(S)
	print(P)
	print(VIN)
	
	z = [not x in Blocked for x in VIN];

	x = numpy.array(S) < 28000 ; # menos de 28000 kilometros recorridos
	y = numpy.array(P) < 380000; # precio del vehuculo sea menos de 380000
	z = numpy.array(z);

	result = numpy.where(x & y & z)[0];
	
	return result;

Blocked = [
'3N6AD33A1LK821729',
'3N6AD33A0LK821883',
'3N6AD35A7LK812434',
'3N6AD35A6LK847353',
'3N6AD33AXJK904203',
'3N6AD33A4KK823330']

while True:
	results = pageProcess();
	#results = []
	if len(results) == 0:
		label = '%s Nothing found trying within 1 hour'%(datetime.datetime.now());
		print(label);
		f = open("log.txt", "a");
		f.write(label + '\n');
		f.close();
		time.sleep(3600);
		continue
	f = open("log.txt", "a");
	f.write('Vehicle found\n');
	f.close();
	print('>>> sending email'); # el agente actua
	os.system('bash email.sh');
	
	break
	
print('[end]')


