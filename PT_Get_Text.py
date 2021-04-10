from NLPyPort.FullPipeline import *

try:
    import spacy
except:
    import spacy

def POS_Portuguese(s, lemmatizer = False, pre_load = True):
    import nltk
    import sklearn
    # Source: https://github.com/NLP-CISUC/NLPyPort
    # Instructions:
    # In case of success, the pipeline will return an object of the “Text” class. 
    # The properties of this are as follow: text.tokens text.pos_tags text.lemas text.entities text.np_tags
    # Additionally, there is a method to return the pipeline in the CoNNL Format: text.print_conll()
    # To separate lines , at the end of each line the additional token EOS is added.
    from IPython.display import clear_output
    clear_output(wait=True)
		
    '''
    Utilização de: https://github.com/NLP-CISUC/NLPyPort
    
    Função que recebe frase ou lista de frases e devolve listas com:
        tokens, lemas (opcional) e POS_tags associadas.
                      
    Parâmetros:
    s >>> frase ou lista de frases,
    lemmatizer (default: False) >>> devolve tokens lematizados.    
    pre_load (default: True) >>> faz load do programa inicialmente.
    '''
    if lemmatizer != True and lemmatizer != False:
        ValueError("Parâmetro lemmatizer dado com valor inválido.")
    
    if pre_load != True and pre_load != False:
        ValueError("Parâmetro pre_load dado com valor inválido.")

    options = {"tokenizer" : True,"pos_tagger" : True,"lemmatizer" : lemmatizer,
               "entity_recognition":False,"np_chunking" : False,
               "pre_load" : pre_load,"string_or_array" : True}
    

    text = new_full_pipe(s,options=options)

    if lemmatizer == False:
        return list(zip(list(text.tokens),list(text.pos_tags)))
    elif lemmatizer == True:
        return list(zip(list(text.tokens),list(text.lemas),list(text.pos_tags)))

def lem_PT(lista):
    lista_POS_PT = POS_Portuguese(lista, lemmatizer = True, pre_load = True)
    return list(set([lista[i][1] for i in range(len(lista)) if lista[i][1] != 'EOS']))

def stem_PT(lista):
    from nltk.stem import RSLPStemmer
    stemmer = nltk.stem.RSLPStemmer()
    return [stemmer.stem(word) for word in lista]

#### Contagem de sílabas
def conta_silabas(palavra, numero_ou_silabas = 'numero'):
    ### Divisor silabico baseado na quase totalidade nas regras do espanhol.
    ### A diferenças em relação ao português são excepções significativamente mais complicadas.
    ### As mais comuns foram adicionadas ao código.
    ### Este divisor para português acerta numa percentagem muito elevada das sílabas (>90%).
    ### Como aquilo que mais me interessa é a contagem das silabas e não a sua divisão a
    ### Precisão do divisor é ainda superior, significativamente perto dos 100%.
    
    ### Original: https://github.com/amunozf/separasilabas
    
    if numero_ou_silabas != 'numero' and numero_ou_silabas != 'silabas':
        raise ValueError("Parâmetro get_sentence_back dado com valor inválido.")
        
    class char():
        def __init__(self):
            pass

    class char_line():
        def __init__(self, word):
            self.word = word
            self.char_line = [(char, self.char_type(char)) for char in word]
            self.type_line = ''.join(chartype for char, chartype in self.char_line)

        def char_type(self, char):
            if char in set(['a', 'á', 'e', 'é', 'ó', 'í', 'ú','o','i']):
                return 'V' #strong vowel
            if char in set(['u', 'ü','ã','õ']):
                return 'v' #week vowel
            if char=='x':
                return 'x'
            if char=='s':
                return 's'
            else:
                return 'c'

        def find(self, finder):
            return self.type_line.find(finder)

        def split(self, pos, where):
            return char_line(self.word[0:pos+where]), char_line(self.word[pos+where:])

        def split_by(self, finder, where):
            split_point = self.find(finder)
            if split_point!=-1:
                chl1, chl2 = self.split(split_point, where)
                return chl1, chl2
            return self, False

        def __str__(self):
            return self.word

        def __repr__(self):
            return repr(self.word)

    class silabizer():
        def __init__(self):
            self.grammar = []

        def split(self, chars):
            rules  = [('VV',1), ('cccc',2), ('xcc',1), ('ccx',2), ('csc',2), ('xc',1), ('cc',1), 
                      ('vcc',2), ('Vcc',2), ('sc',1), ('cs',1),('Vc',1), ('vc',1), ('Vs',1), ('vs',1)]
            for split_rule, where in rules:
                first, second = chars.split_by(split_rule,where)
                if second:
                    if first.type_line in set(['c','s','x','cs']) or second.type_line in set(['c','s','x','cs']):
                        #print 'skip1', first.word, second.word, split_rule, chars.type_line
                        continue
                    if first.type_line[-1]=='c' and second.word[0] in set(['l','r']):
                        continue
                    if first.word[-1]=='l' and second.word[-1]=='l':
                        continue
                    if first.word[-1]=='r' and second.word[-1]=='r':
                        continue
                    if first.word[-1]=='c' and second.word[-1]=='h':
                        continue
                    if first.word[-1]=='c' and second.word[0]=='h': ### added
                        continue
                    if first.word[-1]=='l' and second.word[0]=='h': ### added
                        continue
                    if first.word[-1]=='n' and second.word[0]=='h': ### added
                        continue
    
                        
                    return self.split(first)+self.split(second)
            return [chars]

        def __call__(self, word):
            return self.split(char_line(word))

    silabas = silabizer()
    if numero_ou_silabas == 'numero':
        return len(silabas(palavra.lower()))
    else:
        return silabas(palavra.lower())

def clean_raw_sentence_PT(sentence):

    lista_de_vogais = ['A','E','I','O','U','a','e','i','o',
                   'u','Á','Ã','À','Ä','Â','á','ã','à',
                   'ä','â','É','È','Ë','Ê','é','è','ë',
                   'ê','Í','Ì','Ï','Î','í','ì','ï','î',
                   'Ó','Õ','Ò','Ö','Ô','ó','õ','ò','ö',
                   'ô','Ú','Ù','Ü','Û','ú','ù','ü','û']

    lista_de_consoantes = ['B','C','Ç','D','F','G','H','J','K',
                           'L','M','N','Ñ','P','Q','R','S','T',
                           'V','W','X','Y','Z','b','c','ç','d',
                           'f','g','h','j','k','l','m','n','ñ',
                           'p','q','r','s','t','v','w','x','y','z']

    sentence_listed = sentence.split(' ')

    purged_sentence = []
    for word in sentence_listed:
        purged_word = []
        for char in word:
            if char in lista_de_consoantes or char in lista_de_vogais:
                purged_word.append(char)
        purged_sentence.append(purged_word)
    purged_sentence = ["".join(word) for word in purged_sentence]
    return purged_sentence

def palavras_simples_1000_PT_ES():
    return ["exempl","proib","ren","giraf","amêndo","procur","toqu","pinh","alm","ele","só",
            "aparelh","espír","corp","mud","analis","com","mais","popul",
            "aquil","desej","deix","armazém","public","later","mudanç","arc","per","miléni","liberdad","minor","ingl",
            "di","element","aniversári","unh","cam","entr","amarg","quê","process","sext","prov",
            "diarre","sig","ou","refriger","meu","ide","entend","nível","obrig","tédi","casal",
            "conflit","pic","anoitec","nariz","drag","desenh","apoi","menos","planet","metál",
            "continent","raiz","galáx","tant","noz","menin","solicit","esfaqu","não","políci",
            "esper","sol","aument","amarel","centímetr","nenhum","geral","esp","ar","reproduç",
            "club","esforç","junt","possibil","odei","jaquet","energ","tanch","sistem","ir",
            "cheir","superfíci","joelh","manhã","formal","rir","fic","pergunt","suger","cant",
            "elogi","pel","calm","histór","dist","dor","quent","prat","cad","dúv","mau","educ",
            "em","deterior","caracol","net","desert","dar","comparec","ali","experi","quint","real",
            "pesso","apart","eu","dan","aud","doc","sim","elefant","dent","trig","veícul","borbolet",
            "clar","diári","cart","oitav","encant","vers","aeroport","decis","porqu","imposs",
            "pomb","pássar","equip","fot","ced","combin","font","recent","ampl","regul","cachorr",
            "franc","jorn","result","qual","beb","déc","bom","pergunt","terc","mosc","mas","quant",
            "superi","fresc","interrupç","entreten","com","apreend","desembrulh","qual","import",
            "leã","leit","épic","assist","segund","temp","sement","espanhol","cint","quiet","dobr",
            "ess","entr","aritmé","form","caracterís","estrut","lábi","respond","animal","laranj",
            "cev","vermelh","conceit","pé","marid","esquerd","organiz","grav","muit","total","gá",
            "diagon","rascunh","coraçã","nacion","explic","apen","decret","acompanh","fog","perceç",
            "hav","agit","mesm","dolor","defend","document","canel","ansi","maior","defin","quart",
            "mang","dificuldad","chá","jog","rapid","falcã","local","arroz","físic","japon","ervilh",
            "tranc","benefici","usual","rabanet","inter","dorm","coleg","porc","associ","diferenç",
            "atum","produç","açã","prai","bols","exat","plant","personal","cá","caminh","viv","café",
            "conhec","quadr","me","consequ","ferrov","bast","azul","sucess","our","lag","dicion",
            "oss","recus","human","devag","ministr","obrig","destru","nó","calor","bas","nom","cert",
            "também","fácil","ave","gentil","lua","cotovel","cas","braç","ded","licenç","égu","nov",
            "divid","bilhet","chut","situ","gel","ataqu","forç","escrev","diz","quatr","iss","noss",
            "aul","volum","fl","doming","estil","transport","hor","pat","mund","progress","region",
            "cidad","sair","ond","teu","palm","óbvi","própri","fort","gross","edifíci","pesquis",
            "sens","orelh","idad","dinh","trist","ardent","vestu","papel","express","simpl","paz",
            "hom","atual","assust","acert","faz","congress","cris","govern","ovelh","chanc","geograf",
            "sab","aranh","igual","bisavô","regr","ergu","mosquit","fio","ferr","faculdad","op",
            "decid","cam","rapaz","set","guerr","imediat","natur","tabel","soluç","diret","merc",
            "revist","gram","árvor","quilómetr","part","peix","peit","trâns","ocean","trans","difer",
            "jacaré","prim","ouç","séri","reun","comunic","col","troc","acord","desd","frequ","tomat",
            "castanh","mor","encontr","avali","sáb","seman","janel","salt","horr","símbol","med",
            "grand","pres","sens","gat","dia","praz","peç","ameix","públic","efeit","par","polít",
            "tamanh","conhec","relaç","ato","propós","reg","grip","autor","sobr","carr","colin",
            "sutiã","ros","bigod","bich","veget","ativ","distingu","senh","estranh","cor","notíc",
            "tard","adult","espec","rio","irmã","camp","necess","avanç","já","adolesc","confianç","ambi",
            "eletric","silenci","agor","cavalh","capac","sap","lá","trabalh","divers","pai","art",
            "ment","lagart","frent","compact","apiment","nunc","númer","soc","urban","lul","mar","salg",
            "e","tópic","amig","inset","comum","consider","com","ceno","fim","final","rest","facil",
            "fíg","municípi","nov","ditad","represent","mord","pod","exteri","ont","fadig","atrás","vári",
            "naçã","boc","sai","presid","hotel","larg","real","acord","seguranç","decorr","raiv","destin",
            "inclus","junt","frac","favor","vic","grup","realiz","capaz","fech","anteri","am","segur",
            "rapid","aquel","memór","morr","espec","hoj","feix","códig","sang","univers","est","através",
            "pard","cresc","vend","fábr","cem","gafanhot","unir","relatóri","presenç","dez","atmosf",
            "est","que","cost","front","zer","coloc","até","pos","frang","gol","vest","tal","voz","mod",
            "separ","fals","alguém","serviç","direit","cult","nort","estrel","internac","caval","troc",
            "departament","artig","control","recu","com","contr","fact","interess","precis","descans",
            "entr","pouc","violet","compet","cont","múscul","autoestr","cert","alem","lembr","corv","mãe",
            "signific","empat","ombr","camp","obje","litr","feij","camis","ajud","vid","trê","evit",
            "automá","adicion","concord","termin","tronc","nasc","dur","sem","semelh","sugest","talv","olh",
            "signific","ver","especifiqu","chin","human","tempestad","lei","escrivan","past","pont","paí",
            "fri","barr","doi","apetit","escolh","suav","exist","intenc","bebé","doenç","aproximad","econom",
            "pert","cas","cim","chav","gest","exist","ach","enquant","mim","vend","mostr","oit","traz",
            "ambos","anális","tras","vari","deput","quil","err","chuv","cord","flex","pes","volt","cort",
            "cinz","democrac","admir","montanh","compr","difícil","figur","mê","sociedad","fal","moment",
            "aproxim","quadr","cresc","marisc","trist","jov","milímetr","vis","selv","plan","seil","vent",
            "cinc","centr","person","quas","fas","lug","ser","us","metr","dire","déc","palavr","confort",
            "long","construç","form","irm","beb","mov","molusc","verific","autocarr","vil","fund","semelhanç",
            "tr","carn","rot","objet","cri","interi","boi","substitu","escult","outr","comiss","peg","impost",
            "coelh","priv","folh","direit","rodov","futur","rab","vist","oest","ninguém","fech","deu","pern",
            "complic","livr","estaçã","olh","barb","câmar","aí","bot","ajud","lapel","clim","bomb","autoridad"
            "automó","dad","ent","anteri","mão","branc","qu","seu","direç","compreend","consci","caus",
            "desaparec","constru","conselh","est","tel","estrit","estreit","inic","estômag","interval","cans",
            "elev","criat","govern","surpr","sust","err","centr","kil","milhã","cérebr","segred","sécul",
            "cogumel","filh","encant","nad","verm","preç","prova","destru","sequ","saúd","conex","program",
            "cois","fing","laç","magr","língu","rat","escol","depois","últ","cade","alfabet","sapat","um",
            "probabil","pens","comunic","desport","vap","recolh","alegr","amanhec","bols","atitud","alegr",
            "lament","abaix","calç","vest","comput","mal","o","tud","prim","sent","penúlt","avô","atu",
            "surpreend","compar","tigr","brev","milion","possível","abr","ment","rã","velh","cult","possi",
            "reduç","perceb","calend","fabric","caud","puls","esc","frut","nebul","relat","aceler","prop",
            "corret","institut","músic","par","porém","quadril","lo","voss","cabel","lanç","acredit","etap",
            "aind","entedi","águ","apar","grau","metal","cint","rai","car","fracass","bisavó","jorn",
            "alt","provínc","chumb","cadarç","aqu","separ","escritóri","mei","relig","pior","man","nem","pret",
            "orig","zon","long","incorret","quant","produt","recent","baix","pescoç","mort","aparec","de",
            "ambulânc","lago","fregues","aven","conteúd","afet","capit","mont","cobr","nev","cinem","univers",
            "abdómen","consert","negóci","verdad","univers","companh","lev","rural","dentr","necess","funç",
            "falt","zang","text","non","algum","receb","pêsseg","sobr","entrevist","minut","raç","séri","bambu",
            "delici","artíst","noit","lesm","mil","compr","pot","painel","voc","ano","empr","distrit","afast",
            "instant","desej","porqu","rataz","nádeg","remov","polv","fedorent","terr","por","tent","dificil",
            "florest","espaç","tio","cent","relv","estrang","atenç","sal","espéci","cop","apes","áre","únic",
            "sard","consum","lar","quer","estúdi","set","consegu","trabalh","cheg","trov","visualiz","sét",
            "curs","man","ciênc","ra","reun","port","batat","qu","cas","profiss","cabeç","roup","verd","val",
            "tax","pes","rua","sol","cuec","bisnet","sempr","ter","constitu","famíl","pass","format","agulh",
            "desenvolv","med","cri","qualqu","seguint","neg","maçã","castanh","casac","barat","sul","feliz",
            "excet","melanc","cox","ord","arm","exig","alg","absolut","corr","comunidad","ger","improv",
            "ant","avó"]


def palavras_dale_chall_PT():
    return ["montanh","toc","suculent","ofert","soc","grat","faix","cru","tip","espír","sant","pólv","ler","pisc","tótó",
            "import","pul","calç","nú","praz","chocolat","águ","voo","cóceg","desconhec","cetim","mach","seguranç","que",
            "conduz","dificil","mor","multiplic",
                                      "túlip","combin","seu","ninh","gelé","baix","mendig","barulh","rapid","brev","quest",
            "bebé","aeronav","lamacent","barr","tijol","suav",
                                      "soluç","fat","nível","bandej","temp","arm","corr","salmo","boné","alcanc",
            "trimestr","últ","aproxim","habit","escrit","mãe","serv",
                                      "plataform","rapid","obstácul","riach","excet","autocarr","respir","desmoron",
            "escurec","atir","charlat","lam","vir","cano","nogu",
                                      "receb","colch","esquil","corret","passag","sort","descasc","oitent","peix",
            "embeb","ceg","toqu","bod","selecc","tort","corcund",
                                      "cinz","avent","divid","splash","mergulh","leit","palh","compreend","chocalh",
            "malv","morr","band","vap","pegaj","sap","escorreg","contud",
                                      "cam","desagrad","escond","col","bolach","cas","dev","mult","estud","escolh",
            "desej","excerpt","dificuldad","ferr","pão","aborrec","dan",
                                      "qu","destru","sáb","dard","poder","pert","engenh","pontap","montanh","form",
            "enquant","clar","alpendr","estábul","puls","sufici","raç",
                                      "gráf","humild","péss","cas","apar","grit","caval","desgost","altur","próx",
            "galop","mant","nad","alguém","mar","branc","pratel","march",
                                      "esqu","chal","cotovel","pulg","ganch","senh","cel","surd","laç","noiv","cerc",
            "diab","punh","agulh","tomat","bas","esp","coleg","júni",
                                      "cer","urs","terc","aprove","almoç","indivídu","patif","pap","elogi","procur",
            "cade","mastig","corr","falh","ódi","par","proc","tomb","fé",
                                      "apit","arrulh","vai","praz","arrum","boi","fal","mal","rio","rai","vo","vent",
            "inset","chov","obrig","grau","navi","transport","ferrament",
                                      "cere","hospit","vi","aul","silenci","carruag","examin","est","enferruj",
            "princip","pequenin","coro","autoestr","vest","nariz","sanit",
                                      "inimig","explos","apress","gavet","lament","pec","pom","coc","girassol","mesm",
            "julh","pag","gel","devor","avô","atenç","mens","banc",
                                      "rug","sab","enfraquec","larg","min","unid","castel","chanc","eletric","imens",
            "car","alfinet","líri","can","prov","arrast","gem","caracol",
                                      "garraf","filh","violet","cord","cachecol","desd","gaj","sarn","sof","lâmp","mai",
            "tem","fim","rem","mamã","bic","agir","menos","alarm","dúz",
                                      "pedaç","equip","castanh","títul","vaporiz","nem","leit","espert","pomb","hostil",
            "burr","dent","compr","louc","cabeç","rab","iles","defin",
                                      "lã","do","fund","bronz","portã","corp","oit","recrei","venen","lagart","géme",
            "mudanç","alugu","mam","estaçã","sobrecarg","est","manjedo",
                                      "marfim","mascar","alun","amig","morang","proteg","cop","outubr","pastor","empurr",
            "cicl","pern","feltr","sanduích","corr","infern","esvoaç",
                                      "cafet","satisfatóri","varr","minut","long","frit","sop","cuid","eh","lig",
            "bomb","recompens","pendur","despert","balanç","alt","tol","rádi",
                                      "frit","golf","suport","travess","adivinh","berç","geral","harp","reboqu",
            "abelh","lut","lid","galinh","fl","bal","plant","decid","ali","cabel",
                                      "viag","lua","fad","sempr","guard","caligraf","lat","barb","toalh","piscin",
            "longínqu","novent","shopping","tribun","membr","fund","ligad",
                                      "parapeit","superi","fri","rod","carriç","mistur","preç","sobr","motor",
            "tra","rox","parqu","cozinh","segund","later","gay","vag","masculin",
                                      "cão","háb","sum","trê","objec","visit","cant","buzin","pertenc","extingu",
            "anc","barb","cem","sol","astut","cord","portant","avent","espum",
                                      "frac","mais","suj","atras","pad","obt","apropri","hotel","macac","ge","comprim",
            "palm","molh","afin","lix","escriv","junh","vermelh","agit",
                                      "tent","pérol","limp","linh","teso","relógi","leã","grup","leitelh","preench",
            "cavern","parc","prat","aban","pux","amarel","jov","listr",
                                      "marinh","crianç","trav","saud","martel","bosqu","músic","agost","cint","ram",
            "coç","queix","jog","cheg","sapat","divers","trig","algod",
                                      "empacot","propriet","gat","contabil","lembr","locomo","doc","vesg","abób",
            "aven","buzin","cart","seiv","revist","elefant","saíd","olá","slid",
                                      "casul","aut","cusp","pãe","ocean","encant","pard","trev","bris","dan","alegr",
            "levant","perig","ver","invé","cháven","manivel","doming","poç",
                                      "autor","escol","bêb","poup","relâmpag","que","cham","comport","mass","carvalh",
            "vest","acert","noz","pânt","mostr","febr","lev","traz","quarte",
                                      "peru","espig","tom","cobr","descobr","cas","bochech","junt","vinh","repet",
            "permanec","gasolin","seman","excel","estrel","capaz","estic",
                                      "vagabund","quarent","aprend","gafanhot","fom","piment","prim","cantarol",
            "temper","fog","jacaré","esfome","dez","confortabl","casac","acen",
                                      "bol","cur","hoj","crem","formig","punh","castiç","ar","pretend","ros","nov",
            "visual","unir","bel","dezembr","cal","ambos","aceler","sement",
                                      "rend","centei","cord","possível","cênt","enigm","band","enferm","rainh",
            "context","salt","pent","diz","ment","cober","barc","cogumel",
                                      "estranh","borrif","banh","mão","quatr","confort","ferv","font","uiv","persi",
            "camis","tapet","trap","pis","cor","foguet","coruj","útil",
                                      "disp","relatóri","envi","rach","interrup","nat","ou","dist","ceno","bengal",
            "mant","trez","avis","afi","sujidad","sussurr","trev","incomod",
                                      "cart","folh","boc","acord","chá","cinquent","profes","medic","pintarrox",
            "régu","rod","amizad","pomb","mad","afund","demor","galã",
                                      "através","anj","pacot","poleg","sobrancelh","erv","peg","garr","escorregadi",
            "pneu","trabalh","defês","cart","amanhã","solt","sonolent",
                                      "real","dilet","ombr","pag","invern","brilh","sapat","bord","balanç","ent",
            "ser","heró","notíc","fic","hin","am","aranh","nó","cabin",
                                      "tesour","ecrã","bíbl","filhot","precis","pront","parec","assin","bênç",
            "quiet","gal","elétr","verm","ant","capuz","man","bast","setembr",
                                      "mug","quent","lim","empilh","aren","guerr","farel","amor","escolh","mei",
            "sair","vot","colh","her","rib","uísqu","luv","vend","desesper",
                                      "milhã","vestígi","sin","uniform","mund","difícil","milh","pun","anim","acç",
            "carn","biscoit","descalç","talh","pat","prefer","inici","caroç",
                                      "frut","escur","lob","plan","cebol","selv","vent","bal","escrivan","escav",
            "gargant","rasg","pranch","rebanh","trein","anan","desej","empoeir",
                                      "beb","barulhent","apit","mastr","garf","troc","arvored","alforrec","ra",
            "pessoal","novembr","atrev","gost","estud","quadr","pouc",
                                      "paviment","revé","únic","batalh","just","rei","cinem","cor","águ","lot",
            "cant","escritóri","ladr","mul","mov","pál","desenh","su","mort",
                                      "bale","prai","poleg","sedent","estreit","figur","amig","perdo","apar",
            "pont","papagai","cent","grã","ovelh","lider","banc","vári","golp",
                                      "conch","sol","dur","chinel","bonit","melanc","canal","rap","agradec","porqu",
            "florest","obedec","relut","babet","merc","aeroport","atend",
                                      "pón","espalh","pi","alç","alm","nasc","di","paí","client","brux","derram",
            "compartilh","ser","aconcheg","portugu","diferenç","unh","ah",
                                      "almofad","egoíst","sepult","margar","franz","leit","elev","pássar","cald",
            "túnel","pól","chapéu","lápil","ritm","nenhum","crost","escut",
                                      "mel","admir","papel","pesc","caix","transport","alvor","pul","torc","bolh",
            "pan","real","avis","automó","orelh","ansi","nuv","permit",
                                      "inacab","quer","tron","rein","cair","mold","chup","barril","calafri",
            "perfum","apag","avi","acord","rum","prémi","pen","col","danç","man",
                                      "dobr","amendoim","rap","torn","sest","bol","acamp","direit","bot","gentil",
            "pá","reviravolt","exérc","col","mo","pret","fei","roup","fix",
                                      "aro","e","bat","lebr","remov","cust","linh","fer","fin","luz","superfíci",
            "am","jog","regress","par","tambor","uva","ajud","anc","ondul",
                                      "alfac","dor","assist","sext","av","quant","enevo","pacíf","direç","camp",
            "val","insult","grit","are","simpl","set","relv","zumb","rir",
                                      "ato","ano","sangr","esperanç","encontr","migalh","pr","trist","flor","cacau",
            "mag","not","gel","cov","exat","trov","cortin","mach","alegr",
                                      "embelez","separ","constru","cov","rápid","terr","con","ervilh","curs","brilh",
            "aur","nasc","ferr","númer","jóq","índic","gast","táx",
                                      "convers","lábi","nev","saud","fábr","ded","págin","sucat","verd","pist",
            "ríg","fivel","por","setent","inund","val","orient","pin","visit",
                                      "dançarin","provoc","cisn","maçanet","públic","falcã","dedal","sobretud",
            "interess","pous","sal","municípi","tigr","apert","caç","vel",
                                      "sonh","err","trabalh","sombr","espinafr","congel","foguet","favor","aliment",
            "óle","jorn","com","cerej","mangu","salsich","idad","pass",
                                      "maiorit","desert","ond","cliqu","col","tabel","dezoit","govern","bif","ral",
            "long","fogã","saud","absorv","sec","verdad","paraís","pont",
                                      "penhasc","ar","desfiladeir","morr","contrat","chef","ter","toss","pónil",
            "fest","aba","dent","past","mandíbul","acord","gentil","lu",
                                      "travess","patet","son","paláci","hosped","top","esprem","advog","descrev",
            "troc","lar","cinc","tímid","onz","arbust","paz","etap","orvalh",
                                      "trint","maestr","apen","amarg","empreg","prepar","maldit","panc","ferrov",
            "roch","peg","nev","joelh","rez","pens","sopr","ha","mex","limon",
                                      "contorn","pipoc","bacon","capacet","histór","panquec","culp","us","quinz",
            "cum","pens","pian","esculp","desped","franqu","engraç","deriv",
                                      "set","emprést","emprest","céu","mestr","sujeit","armári","apodrec","fati",
            "dou","dobr","ren","est","chate","tronc","par","nunc","correspond",
                                      "hom","piolh","brind","combust","borbolet","parec","ferrad","sust",
            "reconstru","cigarr","mei","atrás","tigel","ping","cheir","rataz",
                                      "rapaz","hortelã","hor","tal","com","fot","mosc","avali","aço","prédi","cap",
            "tartarug","final","culinár","braç","vivaz","net","noit","outr",
                                      "am","preocup","por","cruel","orgulh","atac","sr","cour","carn","cor",
            "embrulh","fras","irm","sold","animal","gentil","campeã","nerv","flor",
                                      "sai","regist","armadilh","lanç","alegr","jardim","estac","limã","pic","cad",
            "convid","aind","dour","alc","berr","desmai","prad","marc",
                                      "oh","barat","fio","alt","trilh","percevej","palh","rabanet","segu","encaracol",
            "sem","abraç","lavand","palit","marid","instant","colheit",
                                      "ocup","tanqu","ronc","buf","égu","jorn","violin","alcatr","aqu","problemá",
            "rebaix","int","manteig","terr","telefon","interess","afeiço",
                                      "pres","desp","já","transmiss","cacarej","som","porcel","doenç","gab","pobr",
            "art","vil","cl","serviç","fíg","prat","reun","aproximad",
                                      "se","verd","pes","carv","serv","empanturr","ele","quart","um","moed",
            "gabinet","cert","lest","toc","dorminhoc","apart","desej","célul",
                                      "bald","anel","vel","derrot","algum","escal","final","coax","tio","mordidel",
            "dom","pesso","acim","esfom","cont","estal","solareng","caç",
                                      "cidad","segur","fum","gerent","chuv","galh","benv","dól","víg","carr",
            "tord","estr","queim","circ","pic","coxe","coraj","campan","jant",
                                      "dolor","mat","brinqued","descans","natal","vis","metad","mil","aces",
            "dezasset","apart","cascalh","sub","coelh","oss","engol","ave","port",
                                      "negr","biciclet","coloc","fil","poupanç","trenó","catorz","dezassil",
            "regr","prov","espaç","ver","barb","feliz","tod","polac","infânc",
                                      "vaqu","seil","cervej","nort","guard","sent","apost","pel","tempest",
            "tarif","sac","mê","canet","implor","colun","dobr","pescoç","repentin",
                                      "lav","baa","de","também","pass","patim","silênci","ninguém","pijam",
            "flutu","solt","ar","ortograf","repar","bols","lat","peç","fen","cost",
                                      "casc","fever","vez","lasc","manhã","parafus","patin","cort","gar",
            "altifal","oco","delíc","vent","tard","metal","beij","bich","intelig",
                                      "clim","colchã","dinh","velh","parc","almof","caminh","far","trist","banh",
            "gravat","red","tempestad","feij","baralh","segu","olh","viv",
                                      "casac","bolot","vint","conjunt","film","só","sagac","entr","esmag","esfreg",
            "relv","escov","foss","panel","ganh","começ","líd","mau","olh",
                                      "pergunt","fechad","perd","gorjet","injust","final","liberdad","grand",
            "pad","naufrag","gal","tort","mod","test","homossex","argil","rat",
                                      "tricot","comboi","círcul","segur","o","tonel","pedr","depend","compr",
            "comparec","pic","assust","vid","escal","impost","búfal","enfrent",
                                      "tuf","cómod","abaix","cérebr","sul","honest","névo","ovo","adic","nov",
            "fai","bilhet","negóci","saud","sét","alfaiat","agor","pó","azul",
                                      "remend","centr","agricul","chaminé","perig","bande","vantag","tud","junt",
            "tamanh","resolv","dentr","respond","ensin","mont","anc","tras",
                                      "pod","presunt","quint","janel","jan","fad","rebent","corrig","depos",
            "com","caixot","banh","esc","fúr","palavr","falh","co","fich","sens",
                                      "par","talv","cas","pêr","envelop","estrag","refriger","ont","bols",
            "agasalh","bainh","trapac","cidad","café","chamusc","calç","borrach",
                                      "queij","esquerd","est","camp","espirr","pelud","perd","teu","voz","merec",
            "banan","carimb","dirig","musg","fácil","verdade","bibliotec",
                                      "acrescent","elf","solit","princip","bagunç","sob","cans","abr","pint",
            "mármor","rodopi","espess","muit","fresc","public","felic","bot",
                                      "rol","escrev","padr","persegu","gá","corv","bonec","cov","rasg","namor",
            "explic","azed","surg","gui","apresent","piqueniqu","batat","vir",
                                      "canh","tend","mang","rua","teatr","tolic","dad","vas","horr","entr",
            "parent","caus","sim","gargalh","rabugent","fals","capit","costur",
                                      "prend","fech","gigant","moment","apont","gel","acontec","sortud","sel",
            "ping","poç","test","lag","cort","possi","sombri","entr","emb",
                                      "pé","meu","difer","desenh","imposs","mim","rasp","jub","gord","pai",
            "desdobr","lagart","apress","couraç","livr","caramel","bondad","nubl",
                                      "desliz","lev","sobrelot","avó","rot","fer","fortun","trat","rastej",
            "porc","barr","guardanap","selim","choqu","groselh","prim","viv",
                                      "motiv","escotilh","quas","miner","tub","pôr","infeliz","cozed","med",
            "esper","color","dic","taref","preguiç","puzzl","plan","níquel",
                                      "salgu","geograf","lenç","banc","tare","ajoelh","bag","perdiz","gangu",
            "motor","sal","pop","anã","aceit","sofr","past","cel","jarr",
                                      "libert","consert","govern","coraçã","doçur","oitav","íngrem","carreg",
            "assopr","mobíl","tec","feri","chicot","apanh","doc","gom","cint",
                                      "escap","glór","lug","repolh","vitór","tabac","ir","armazen","aquec",
            "colin","companh","fur","sup","ment","moribund","velej","rol","ajust",
                                      "escur","gad","ced","furi","tu","sent","promet","faz","ketchup","cemitéri",
            "vali","dia","març","borr","profund","espanhol","fogu","doz",
                                      "cont","traç","orden","escut","sagr","ve","serr","forn","igrej","signific",
            "bacalhau","banh","cont","quadr","tricicl","doi","laranj",
                                      "crem","sofr","própri","segred","trag","capitã","telh","mergulh","confi",
            "pequen","local","antiqu","prejuíz","educ","portug","irmã","qualqu",
                                      "ferrug","fit","toranj","encost","palh","raiz","maçã","ilh","descu",
            "tamp","calend","perd","cortiç","até","deix","tosqui","melhor",
                                      "madr","frent","vend","encontr","colme","ameix","melr","moinh","abelh",
            "vergonh","sed","fund","fal","mas","braç","cart","fort","rang","poem",
                                      "nov","asa","baix","entusiasm","cólic","reembols","cavalh","esquec",
            "agrad","abenço","caíd","maravilh","palavr","agricult","tour","cuidad",
                                      "cóp","batiz","livr","cobr","futebol","cav","artist","noss","concelh",
            "húm","patrã","complet","lei","arc","encant","ocident","vig","bem",
                                      "nebul","rei","vidr","igual","camel","compar","curv","concord","medicin",
            "caval","qu","pinh","mi","beterrab","cresc","escrav","acid","veloz",
                                      "saúd","bezerr","cois","pris","ide","inch","diret","suspir","bigod",
            "lantern","embarc","aud","quê","outon","língu","mar","gorr","tint",
                                      "roub","fraqu","recus","veget","cest","bec","defend","deu","desperdiç",
            "podr","com","sang","valentim","fend","feitiç","frigide","beisebol",
                                      "despach","esperanç","box","morceg","arroz","carpint","degrau","verific",
            "dar","golf","danç","complet","peit","miau","famíl","milh","hardw",
                                      "volt","bom","aniversári","floc","costur","ardós","potr","club","esp",
            "conhec","devag","burac","esbelt","part","apanh","cas","costel",
                                      "ladr","ess","dur","dobr","desport","açúc","insect","tet","lamb","ond",
            "fábul","cert","não","velud","maquin","map","mir","celebr","vésp",
                                      "lad","doent","pur","fig","ass","mult","empat","fac","em","sei","surpr",
            "alegr","faculdad","canec","engan","descuid","máquin","sábi",
                                      "políci","enterr","impress","primav","madur","desculp","eu","debaix",
            "ranch","ténil","empr","fix","respir","pistol","órg","molh","cade",
                                      "nom","our","caud","trabalh","pêsseg","arej","press","vazi","beir",
            "mord","quebr","pot","sulist","alg","baí","gaiol","salt","caranguej",
                                      "afast","adeu","caban","chor","oraç","pared","chav","ouv","liç","ajud",
            "som","espelh","ideal","flux","honr","esp","quietud","ric","list",
                                      "cruz","espre","estr","oest","gans","afog","sac","can","juiz","lanç",
            "aplaud","lent","assassinat","dispar","arqu","cachorr","ampl","ai",
                                      "diam","and","contr","maravilh","depois","nad","err","refe","porc",
            "dam","enx","vasso","pontu","aeródr","sorris","cegonh","abril","vizinhanç",
                                      "qual","acredit","giz","venc","gost","import","árvor","cam","torr",
            "libr","ass","port","dezanov","extr","lacticíni","di"]

def clean_chars_PT(lista):
    lista_de_vogais = ['A','E','I','O','U','a','e','i','o',
                       'u','Á','Ã','À','Ä','Â','á','ã','à',
                       'ä','â','É','È','Ë','Ê','é','è','ë',
                       'ê','Í','Ì','Ï','Î','í','ì','ï','î',
                       'Ó','Õ','Ò','Ö','Ô','ó','õ','ò','ö',
                       'ô','Ú','Ù','Ü','Û','ú','ù','ü','û']

    lista_de_consoantes = ['B','C','Ç','D','F','G','H','J','K',
                           'L','M','N','Ñ','P','Q','R','S','T',
                           'V','W','X','Y','Z','b','c','ç','d',
                           'f','g','h','j','k','l','m','n','ñ',
                           'p','q','r','s','t','v','w','x','y','z']
    lista_output = []
    for palavra in lista:
        delete = False
        for letra in palavra:
            if letra not in lista_de_vogais and letra not in lista_de_consoantes:
                delete = True
        if delete != True:
            lista_output.append(palavra)
            
    return lista_output

def get_sentence_info(sentence, get_sentence_back = False, get_headings = False, just_headings = False):
    import numpy as np
    # << These are the headings of the output >>
    sentence_headings = ['Word_C','Verb_C','Noun_C','Adj_C','Other_C','Sum_W_Len','Num_Word_Len_Less_4',
                         'Num_Syl','Simp_Word_1','Num_PolySyl','Num_MonoSyl','Simp_Word_DC']

    if get_sentence_back != True and get_sentence_back != False: 
        # Made a way to get the headings without anything else, was useful in development.
        raise ValueError("Parâmetro just_headings dado com valor inválido.")
    elif just_headings == True:
        return sentence_headings
    else: 
        # Getting a raw sentence with only valid characters in list format.
        clean_raw_sentence = clean_raw_sentence_PT(sentence)

        # << Getting sentence info via POS_Portuguese >>
        POS_Info = POS_Portuguese(sentence, lemmatizer = True)

        # << Cleaning the POS_Portuguese output obtained>> 
        templist1 = [tup for tup in POS_Info if tup[1] != 'EOS' and tup[1] != 'punc']    
        punct_symbols_list = ["!","#","$","%","&","(",")","`","´","*","+","-",".","/",":",";","^",
                              "<","=",">","?","@",",","{","_","|","}","~","€","“","”",'"',"'"]
        punct_symbols_list.append(',')
        punct_symbols_list.append('[')
        punct_symbols_list.append(']')
        punct_symbols_list.append('\\')
        templist2 = []
        for tup in templist1:
            remove = False
            for char in tup[0]:
                if char in punct_symbols_list:
                    remove = True
                    break
            if not remove:
                templist2.append(tup)
        words = [tup[0] for tup in templist2]
        lem = [tup[1] for tup in templist2]
        pos_normal = [tup[2] for tup in templist2]
        pos_simple = [tup[2].split('-')[0] for tup in templist2]  

        clean_POS_Info = list(zip(words,pos_normal,pos_simple,lem))

        # << Getting the Sentences Caracteristics >>

        sentence_characteristics = []
        # sentence_chars[0] -> Number of words of sentence.
        sentence_characteristics.append(len(clean_raw_sentence))
        # sentence_chars[1] -> Number of verbs on sentence.
        sentence_characteristics.append(np.sum([tup[2].lower()=='v' for tup in clean_POS_Info]))
        # sentence_chars[2] -> Number of nouns in sentence.
        sentence_characteristics.append(np.sum([tup[2].lower()=='n' for tup in clean_POS_Info]))
        # sentence_chars[3] -> Number of adjectives in sentence.
        sentence_characteristics.append(np.sum([tup[2].lower()=='adj' for tup in clean_POS_Info]))    
        # sentence_chars[4] -> Number of non-verbs, non-nouns, non-adjectives in sentence.
        sentence_characteristics.append(len(clean_raw_sentence) - np.sum([tup[2].lower()=='n' for tup in clean_POS_Info]) - np.sum([tup[2].lower()=='v' for tup in clean_POS_Info]) - np.sum([tup[2].lower()=='adj' for tup in clean_POS_Info])) 
#       sentence_characteristics.append(np.sum([tup[2].lower()!='adj' and tup[2].lower()!='n' and tup[2].lower()!='v' for tup in clean_POS_Info]))    
        # sentence_chars[5] -> Sum of word lenght.
        sentence_characteristics.append(np.sum([len(word) for word in clean_raw_sentence]))
        # sentence_chars[6] -> Number of words with size <= 3.
        sentence_characteristics.append(np.sum([len(word) <= 3 for word in clean_raw_sentence]))
        # sentence_chars[7] -> Number of syllables:
        sentence_characteristics.append(np.sum([conta_silabas(word) for word in clean_raw_sentence_PT(sentence)]))
        # sentence_chars[8] -> Number of easy words 1000 words ES [comparing stemmed lemas]:
        sentence_characteristics.append(np.sum([word in palavras_simples_1000_PT_ES() for word in clean_chars_PT(stem_PT([tup[3] for tup in clean_POS_Info]))]))
        # sentence_chars[9] -> Number of polysyllables
        sentence_characteristics.append(np.sum([conta_silabas(word) > 1 for word in clean_raw_sentence_PT(sentence)]))
        # sentence_chars[10] -> Number of monosyllables
        sentence_characteristics.append(np.sum([conta_silabas(word) == 1 for word in clean_raw_sentence_PT(sentence)]))
        # sentence_chars[11] -> Number of easy words Dale Chall words [comparing stemmed lemas]:
        sentence_characteristics.append(np.sum([word in palavras_dale_chall_PT() for word in clean_chars_PT(stem_PT([tup[3] for tup in clean_POS_Info]))]))
        
        # << Returning Output >>
        if get_sentence_back != True and get_sentence_back != False:
            raise ValueError("Parâmetro get_sentence_back dado com valor inválido.")
        elif get_sentence_back == True: ### The sentence being return is the treated one.
            if get_headings != True and get_headings != False:
                raise ValueError("Parâmetro get_headings dado com valor inválido.")
            elif get_headings == True:
                sentence_rebuilt = rebuild_sentence([tup[0] for tup in templist3])
                return [sentence_rebuilt,list(zip(sentence_headings,sentence_characteristics))]  
            else:
                sentence_rebuilt = rebuild_sentence([tup[0] for tup in templist3])
                return [sentence_rebuilt,sentence_characteristics]  
        elif get_sentence_back == False:
            if get_headings != True and get_headings != False:
                raise ValueError("Parâmetro get_headings dado com valor inválido.")
            elif get_headings == True:
                return list(zip(sentence_headings,sentence_characteristics))
            else:
                return sentence_characteristics
            
def estrutura_sintatica_frase(dependencies_tags, pred_tags, dep_words,lemmas):
    import numpy as np
    # Adaptado e simplificado da função com mesmo nome dentro do projeto: https://github.com/mattgoncalves/PE2LGP
    """
    Define a ordem frásica da frase em português (ex: "SVO") com base nas tags do SpaCy.
    :param dependencies_tags: lista com as etiquetas de dependencia.
    :param pred_tags: lista com as etiquetas morfossintáticas.
    :param dep_words: lista com as palavras da frase.
    :return: uma string com a estrutura frásica da frase (ex: "SVO") e os indices do sujeito, do verbo e objeto, na frase.
    """

    estrutura = []
    indice_verbo = -1
    indice_subj = -1
    indice_obj = -1

    for index, item in enumerate(dependencies_tags):
        if "nsubj" in item:
            estrutura.append("S")
            indice_subj = index
        if "obj" in item:
            estrutura.append("O")
            indice_obj = index
        if "amod" in item and pred_tags[index].startswith("V") and "ROOT" in dependencies_tags:
            estrutura.append("V")
            indice_verbo = index
        if "ROOT" in item and "cop" not in dependencies_tags and pred_tags[index].startswith("V"):
            estrutura.append("V")
            indice_verbo = index
        if "ROOT" in dependencies_tags and pred_tags[index].startswith("V"):
            estrutura.append("V")
            indice_verbo = index
        if "cop" in item and "ROOT" in dependencies_tags and pred_tags[index].startswith("V"):
            estrutura.append("V")
            indice_verbo = index
        if "ROOT" in item and "cop" in dependencies_tags: ###é o caso do predicativo do sujeito
            estrutura.append("V")

    estrutura = list(dict.fromkeys(estrutura))

    if indice_subj != -1 and indice_verbo != -1 and indice_obj != -1: ### SVO
        temp = "".join(estrutura), lemmas[indice_subj], lemmas[indice_verbo], lemmas[indice_obj]
    elif indice_subj != -1 and indice_verbo != -1 and indice_obj == -1: ### SV
        temp = "".join(estrutura), lemmas[indice_subj], lemmas[indice_verbo], np.nan
    elif indice_subj != -1 and indice_verbo == -1 and indice_obj != -1: ### SO
        temp = "".join(estrutura), lemmas[indice_subj], np.nan, lemmas[indice_obj]
    elif indice_subj != -1 and indice_verbo == -1 and indice_obj != -1: ### VO
        temp = "".join(estrutura), np.nan, lemmas[indice_verbo], lemmas[indice_obj]
    elif indice_subj != -1 and indice_verbo == -1 and indice_obj != -1: ### S
        temp = "".join(estrutura), lemmas[indice_subj], np.nan, np.nan
    elif indice_subj != -1 and indice_verbo == -1 and indice_obj != -1: ### V
        temp = "".join(estrutura), np.nan, lemmas[indice_verbo], np.nan
    elif indice_subj != -1 and indice_verbo == -1 and indice_obj != -1: ### 0
        temp = "".join(estrutura), np.nan, np.nan, lemmas[indice_obj]
    else:
        temp = [np.nan,np.nan,np.nan,np.nan]
    
    output = [np.nan if x==" " else x for x in temp]
    
    if output[1] == np.nan and "S" in output[0]:
        output[0] = output[0].replace("S","")
        
    if output[2] == np.nan and "V" in output[0]:
        output[0] = output[0].replace("V","")

    if output[3] == np.nan and "O" in output[0]:
        output[0] = output[0].replace("O","")
        
    return temp

def clean_raw_sentence_PT(sentence):

    lista_de_vogais = ['A','E','I','O','U','a','e','i','o',
                   'u','Á','Ã','À','Ä','Â','á','ã','à',
                   'ä','â','É','È','Ë','Ê','é','è','ë',
                   'ê','Í','Ì','Ï','Î','í','ì','ï','î',
                   'Ó','Õ','Ò','Ö','Ô','ó','õ','ò','ö',
                   'ô','Ú','Ù','Ü','Û','ú','ù','ü','û']

    lista_de_consoantes = ['B','C','Ç','D','F','G','H','J','K',
                           'L','M','N','Ñ','P','Q','R','S','T',
                           'V','W','X','Y','Z','b','c','ç','d',
                           'f','g','h','j','k','l','m','n','ñ',
                           'p','q','r','s','t','v','w','x','y','z']

    sentence_listed = sentence.split(' ')

    purged_sentence = []
    for word in sentence_listed:
        purged_word = []
        for char in word:
            if char in lista_de_consoantes or char in lista_de_vogais:
                purged_word.append(char)
        purged_sentence.append(purged_word)
    purged_sentence = ["".join(word) for word in purged_sentence]
    return purged_sentence

def SVOed_Sentences(Sentences):
    import spacy
    nlp = spacy.load("pt_core_news_sm")
    output = []
    for sentence in Sentences:
        doc = nlp(" ".join(clean_raw_sentence_PT(sentence)))
        dep_words = [token.text for token in doc]
        pred_tags = [token.tag_ for token in doc]
        dependencies_tags = [token.dep_ for token in doc]
        lemmas = [token.lemma_ for token in doc]
        output.append(estrutura_sintatica_frase(dependencies_tags, pred_tags, dep_words, lemmas))
    return output

def count_seq_TRUE(alist): #### Counts: [True, True, True, False] as one   
    counter = 0
    lenght = len(alist)
    if lenght <= 10:
        return np.nan
    else:
        for i in range(lenght-2):
            if alist[i] == True and alist[i+1] == True and (alist[i+2] == False):
                counter += 1
        if alist[-1] == True and alist[-2] == True:
            counter += 1
        return counter

def get_pairs_of_subsequent_TRUE(alist): #### Counts: [True, True, True, False] as two
    counter = 0
    lenght = len(alist)
    sentences_pairs = []
    for i in range(lenght-1):
        if alist[i] == True and alist[i+1] == True:
            sentences_pairs.append((i,i+1))
    return sentences_pairs

def SVO_elements_in_row(Info_Sentences_SVO,Index_Pares_Sentences_SVO):
    import pandas as pd
    countador = 0
    for index_par in Index_Pares_Sentences_SVO:
        if Info_Sentences_SVO[index_par[0]][1] == Info_Sentences_SVO[index_par[1]][1] and pd.notnull(Info_Sentences_SVO[index_par[0]][1]):
            countador += 1
        if Info_Sentences_SVO[index_par[0]][2] == Info_Sentences_SVO[index_par[1]][2] and pd.notnull(Info_Sentences_SVO[index_par[0]][2]):
            countador += 1
        if Info_Sentences_SVO[index_par[0]][3] == Info_Sentences_SVO[index_par[1]][3] and pd.notnull(Info_Sentences_SVO[index_par[0]][3]):
            countador += 1
    return countador

# def count_SVO_found(Sentences):
#     return np.sum([pd.notnull(row[0]) for row in SVOed_Sentences(Sentences)])

# def count_SVO_found_in_a_row(Sentences):
#     return count_seq_TRUE([pd.notnull(row[0]) for row in SVOed_Sentences(Sentences)])
    
# def count_SVO_elements_in_a_row(Sentences):
#     return SVO_elements_in_row(SVOed_Sentences(Sentences),get_pairs_of_subsequent_TRUE([pd.notnull(row[0]) for row in SVOed_Sentences(Sentences)]))


def hh_index_on_text(mytext):
    import pandas as pd
    import nltk
    stopwords = nltk.corpus.stopwords.words('portuguese')

    word_count = len([word for word in clean_raw_sentence_PT(mytext.lower())])
    my_series = pd.Series([word for word in clean_raw_sentence_PT(mytext.lower()) if word not in stopwords or word not in ['',' ','o','a','e']]).value_counts().reset_index()
    my_series = my_series[my_series['index'] != ''].rename(columns={"index": "Word", 0: "Freq"}).reset_index().drop('index', axis = 1)
    my_series['freq_percent_sq'] = (my_series['Freq'] / np.sum(my_series['Freq']))**2
    uniques_per_word = len(my_series)/word_count

    hh_index = np.sum(my_series['freq_percent_sq'])    
    return uniques_per_word, hh_index   
def sentence_split(text):
    import nltk
    raw_text = text.replace('\n',' ')
    raw_text = raw_text.replace('    ',' ')
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    list_of_sentences = sent_tokenizer.tokenize(raw_text)
    return list_of_sentences

def Get_All_Text_Info(text):
    
    ### << PRE-PARSING >>    
    list_of_sentences                    = sentence_split(text)
    SVOed_Sentences_S                    = SVOed_Sentences(list_of_sentences)
    sentence_count                       = len(list_of_sentences)
    
    text_info = []
    for sentence in list_of_sentences:
        sentence_info                    = get_sentence_info(sentence, 
                                                             get_sentence_back = False,
                                                             get_headings = False,
                                                             just_headings = False)
        text_info.append(sentence_info)
    
    word_count = np.sum([sent_info[0] for sent_info in text_info])
    
    ### << Features >> 
    
    # Elementos de dimensão
    Words_per_sentence                   = word_count / sentence_count
    Syllables_per_word                   = np.sum([sent_info[7] for sent_info in text_info])  / word_count 
    Poly_Syl_per_word                    = np.sum([sent_info[9] for sent_info in text_info])  / word_count
    
    # Elementos de análise morfológica     
    Verbs_per_word                       = np.sum([sent_info[1] for sent_info in text_info])  / word_count 
    Noun_per_word                        = np.sum([sent_info[2] for sent_info in text_info])  / word_count 
    Adj_per_word                         = np.sum([sent_info[3] for sent_info in text_info])  / word_count 

    # Elementos léxicais
    Simp_Word_DC_per_word                = np.sum([sent_info[11] for sent_info in text_info]) / word_count 
    Simp_Word_1_per_word                 = np.sum([sent_info[8] for sent_info in text_info])  / word_count
    Uniques_per_word                     = hh_index_on_text(text)[0]
    HH_index                             = hh_index_on_text(text)[1]
    
    # Elementos de análise sintática
    count_SVO_found_per_word             = np.sum([pd.notnull(row[0]) for row in SVOed_Sentences_S]) / word_count
    count_SVO_found_in_a_row_per_word    = count_seq_TRUE([pd.notnull(row[0]) for row in SVOed_Sentences_S]) / word_count
    
    if pd.isnull(count_SVO_found_in_a_row_per_word) == True:
        count_SVO_elements_in_a_row_per_word = 0
    
    count_SVO_elements_in_a_row_per_word = SVO_elements_in_row(SVOed_Sentences_S,get_pairs_of_subsequent_TRUE(
                                           [pd.notnull(row[0]) for row in SVOed_Sentences_S])) / word_count
    
    if pd.isnull(count_SVO_elements_in_a_row_per_word) == True:
        count_SVO_elements_in_a_row_per_word = 0
        
    
    temp = [Words_per_sentence,
              Syllables_per_word,
              Poly_Syl_per_word,
              Verbs_per_word,
              Noun_per_word,
              Adj_per_word,
              Simp_Word_DC_per_word,
              Simp_Word_1_per_word,
              Uniques_per_word,
              HH_index,
              count_SVO_found_per_word,
              count_SVO_found_in_a_row_per_word,
              count_SVO_elements_in_a_row_per_word]
    
    output = []
    for element in temp:
        if pd.isnull(element) == True:
            output.append(0.0)
        else:
            output.append(element)
    
    return output