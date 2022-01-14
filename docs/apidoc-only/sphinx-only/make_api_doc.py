#!/usr/bin/env python3
'''This is a executable python script to build the ECLAIRs documentation.

The script needs in argument the path of the folder to build the doc in.

'''

import argparse
import os
import logging
import subprocess


def get_root_pkg():
    return os.getenv("GRAND_ROOT")

    

def main(args):
    """This is the main function of the executable python script to build the documentation of the ECLAIRs pipeline code.

    :param args: argument parser
    :type args: parse_args
    """
    cmd = "git rev-parse --abbrev-ref HEAD"
    branch = subprocess.getoutput(cmd)
    os.environ["GIT_BRANCH"] = branch
    
    logging.info('########################## ECLAIRs documentation builder ##########################')
    
    doc_dir = os.path.abspath(args.doc_dir)
    logging.info(f"============> [1/4] create/clean {doc_dir} for sphinx-apidoc")
    os.system(f'rm -rf {doc_dir}') 
    try:
        #os.makedirs(doc_dir)
        #os.makedirs(doc_dir + '/src')
        os.makedirs(doc_dir + '/src')
    except FileExistsError:
        pass
    #                        
    logging.info(f"============> [2/4] Copy conf file for API doc and source")
    # Note : get_root_eclairs() has changed
    os.system('cp -r ' + get_root_pkg() + '/docs/apidoc-only/sphinx ' + doc_dir)
    cp_src = 'cp -r ' + get_root_pkg() + '/grand'
    if args.only is None:
        cp_src += ' ' + doc_dir + '/src'            
    else:
        cp_src += f'/{args.only} {doc_dir}/src'        
    print(cp_src)
    os.system(cp_src)    
    current_dir = os.getcwd()
    os.chdir(doc_dir + '/sphinx')
    if args.skip_uml:
        cmd = 'mv source/_templates/module.rst source/_templates/module_UML.rst'
        print(f'cmd: {cmd}')
        os.system(cmd)        
        cmd = 'mv source/_templates/module_noUML.rst source/_templates/module.rst'
        print(f'cmd: {cmd}')
        os.system(cmd)        
       
    logging.info(f"============> [3/4] sphinx-apidoc with better-apidoc")
    os.system(f'sphinx-apidoc -ef -d 7 -o source {doc_dir}/src/grand')    
    logging.info(f"============> [4/4] Do make html")
    os.system('make html')
    os.chdir(current_dir)
    logging.info(f"Documentation in {doc_dir}")
    logging.info('########################## ECLAIRs Documentation builder end #######################')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='executable python script to build the ECLAIRs documentation',
        epilog="""This is a executable python script to build the ECLAIRs documentation.
        The script needs in argument the path of the folder to build the doc in.
        It usesthe Python Documentation Generator SPHNX.""")
    parser.add_argument(
        'doc_dir',
        help="""directory to build the documentation in""")
    parser.add_argument('--skip_uml', default=False, action='store_true')
    parser.add_argument('--only')
    
    args = parser.parse_args()
    
    logging.basicConfig(format='%(message)s', level=logging.INFO)
        
    main(args)
