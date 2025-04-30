# -*- coding: utf-8 -*-
"""
Autor: Juan Ignacio Cerrudo
Fecha: Febrero 2025
"""
import re

def process_classifier(code, type_clf):
    """
    ------------------------
    INPUT:
    --------
    code: array con el código generado del modelo en C++
    type_clf : tipo de clasificador:
        -RandomForest
        -GaussianNB
    ------------------------
    OUTPUT:
    --------
    Archivo procesado en ANSI C    
    """
    with open("Classifier.c", "w") as file:
        file.write('#include "Classifier.h"\n\n')
    
    x = code.split("protected:")
    index_public = x[0].find("public:")
    index_protected = x[1].find("protected:")
    index_protected_end = x[1].find("};")
    code = x[0][index_public+7:]
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};':
            indent -= 1
        if line[-1] == '}' and indent == 0:
            pretty.append('};')
            #continue
        else:
            pretty.append(('    ' * indent) + line)
        
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:':
            indent += 1
            
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty) 
    with open("Classifier.c", "a") as file:
        file.write(pretty)
    #pprint(pretty)
    code = x[1][index_protected+10:index_protected_end]
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};':
            indent -= 1
        if line[-1] == '}' and indent == 0:
            pretty.append('};')   
        else:    
            pretty.append(('    ' * indent) + line)
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:':
            indent += 1
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty)       
    #print(pretty)    
    with open("Classifier.c", "a") as file:
        file.write(pretty)
    with open("Classifier.h", "w") as file:
        file.write('#ifndef _INC_MODEL_H_\n')    
        file.write('#define _INC_MODEL_H_\n\n')
        file.write('#include <stdarg.h>\n')
        file.write('#include <stdint.h>\n\n')
        file.write('int predict(float *x);\n')
        file.write('const char* idxToLabel(uint8_t classIdx);\n')
        file.write('const char* predictLabel(float *x);\n')
        if type_clf == 'GaussianNB':
            file.write('float gauss(float *x, float *theta, float *sigma);\n\n')
        if type_clf == 'RandomForest' or type_clf == 'SVM':
            file.write('float compute_kernel(float *x, ...);\n\n')
        file.write('#endif /* _INC_MODEL_H_ */')
        
        
def proccess_window(code):
    
    with open("Window.c", "w") as file:
        file.write('#include "Window.h"\n\n')
    
    x = code.split("protected:")
    index_public = x[0].find("public:")
    index_protected = x[1].find("protected:")
    index_protected_end = x[1].find("};")
    
    code = x[1][index_protected+10:index_protected_end]
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};':
            indent -= 1
        if line[-1] == '}' and indent == 0:
            pretty.append('};')   
        else:    
            pretty.append(('    ' * indent) + line)
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:':
            indent += 1
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    pretty = re.sub(r'= NULL', '', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty)      
    with open("Window.c", "a") as file:
        file.write(pretty)
        file.write('\n') 
    #pprint(pretty)    
    code = x[0][index_public+7:]
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};':
            indent -= 1
        if line[-1] == '}' and indent == 0:
            pretty.append('};')
            #continue
        else:
            pretty.append(('    ' * indent) + line)
        
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:':
            indent += 1
            
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty) 
    with open("Window.c", "a") as file:
        file.write(pretty)
    #print(pretty)
    
    
    with open("Window.h", "w") as file:
        file.write('#ifndef _INC_WINDOW_H_\n')    
        file.write('#define _INC_WINDOW_H_\n\n')
        file.write('#include <stdarg.h>\n')
        file.write('#include <stdint.h>\n\n')
        file.write('#include "lpc_types.h"\n\n')
        file.write('#include "math.h"\n\n')
        file.write('#define abs(a) (((a) < (0)) ? (-a) : (a))\n\n\n')
        
        file.write('bool transform(float *x, float *dest);\n')
        file.write('void clear(void);\n')
        file.write('#endif /* _INC_WINDOW_H_ */')        
    file1 = open("Window.c", "r")   
    # Complicated way of moving features vector to Window.h
    string1 = "float features"
    string2 = "bool transform"
    index1 = 0
    index2 = 0
    for line in file1:
        if string1 in line:
            index1 = line
            #pprint(index1)
        if string2 in line:
            index2 = line
            #pprint(index2)  
    with open("Window.c", "r") as fp:
        lines = fp.readlines()

    with open("Window.c", "w") as fp:
        for line in lines:
            if line != index1:           
                if line == index2:  
                    fp.write("bool transform(float *x, float *dest){\n")
                else:
                    fp.write(line)
            else:
                print("features vector removed")
    lines = open("Window.h", 'r').readlines()
    lines[11] = index1
    out = open("Window.h", 'w')
    out.writelines(lines)
    out.close()
    
def proccess_pca(code):
    """
    ------------------------
    INPUT:
    --------
    code: array con el código generado del modelo en C++
    ------------------------
    OUTPUT:
    --------
    Archivo procesado en ANSI C    
    """
    with open("Pca.c", "w") as file:
        file.write('#include "Pca.h"\n\n')
    
    x = code.split("protected:")
    index_public = x[0].find("public:")
    index_protected = x[1].find("protected:")
    index_protected_end = x[1].find("};")
    
    code = x[1][index_protected+10:index_protected_end]
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};':
            indent -= 1
        if line[-1] == '}' and indent == 0:
            pretty.append('};')   
        else:    
            pretty.append(('    ' * indent) + line)
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:':
            indent += 1
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    pretty = re.sub(r'= NULL', '', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty)      
    with open("Pca.c", "a") as file:
        file.write(pretty)
        file.write('\n') 
    #pprint(pretty)    
    code = x[0][index_public+7:]
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};':
            indent -= 1
        if line[-1] == '}' and indent == 0:
            pretty.append('};')
            #continue
        else:
            pretty.append(('    ' * indent) + line)
        
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:':
            indent += 1
            
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty) 
    with open("Pca.c", "a") as file:
        file.write(pretty)
    #print(pretty)
    
    
    with open("Pca.h", "w") as file:
        file.write('#ifndef _INC_PCA_H_\n')    
        file.write('#define _INC_PCA_H_\n\n')
        file.write('#include <stdarg.h>\n')
        file.write('#include <stdint.h>\n\n')
        file.write('#include "lpc_types.h"\n\n')
        file.write('#include "math.h"\n\n')
        #file.write('#define abs(a) (((a) < (0)) ? (-a) : (a))\n\n\n')
        
        file.write('bool transform(float *x, float *dest);\n')
        file.write('void dot(float *x, ...);\n')
        file.write('#endif /* _INC_PCA_H_ */')        


def iir_sos_header(fname_out, SOS_mat):
    """
    Write IIR SOS Header Files
    File format is compatible with ESP-DSP IIR 
    Directform II Filter Functions
        
    INPUT:
    --------
    fname_out: nombre del archivo
    SOS_mat: matriz SOS de coeficientes
    ------------------------
    OUTPUT:
    --------
    Archivo procesado *.h    
    """
    Ns, Mcol = SOS_mat.shape
    f = open(fname_out, 'wt')
    f.write('//define a IIR SOS CMSIS-DSP coefficient array\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write('#ifndef STAGES\n')
    f.write('#define STAGES %d\n' % Ns)
    f.write('#endif\n')
    f.write('/*********************************************************/\n');
    f.write('/*                     IIR SOS Filter Coefficients       */\n');
    f.write('float ba_coeff[%d] = { //b0,b1,b2,a1,a2,... by stage\n' % (5 * Ns))
    for k in range(Ns):
        if (k < Ns - 1):
            f.write('    %+-13e, %+-13e, %+-13e,\n' % \
                    (SOS_mat[k, 0], SOS_mat[k, 1], SOS_mat[k, 2]))
            f.write('    %+-13e, %+-13e,\n' % \
                    (SOS_mat[k, 4], SOS_mat[k, 5]))
        else:
            f.write('    %+-13e, %+-13e, %+-13e,\n' % \
                    (SOS_mat[k, 0], SOS_mat[k, 1], SOS_mat[k, 2]))
            f.write('    %+-13e, %+-13e\n' % \
                    (SOS_mat[k, 4], SOS_mat[k, 5]))
    f.write('};\n')
    f.write('/*********************************************************/\n')
    f.close() 