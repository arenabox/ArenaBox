## About
# This script can be used to convert pdf files to xml using [Grobid](https://github.com/kermitt2/grobid_client_python). We used this to convert scientific articles to xml format, which can later be processed to extract
# relevant information.

# Run the command directly on shell or run this file. Change value for --input parameter as folder containing pdf files and change value for --output parameter as folder where you want to save xml files.

grobid_client --input "PATH/TO/PDF/FILES/" --output "PATH/TO/SAVE/XML/FILES/" processFulltextDocument

# For our dataset we used following commands:
# 1. Combined 217 Corpus :
# grobid_client --input "pdfs/217_Corpus_STRAND/217 Corpus/" --output "xmls/217 Corpus/" processFulltextDocument
# runtime: 142.811 seconds

# 2. EIST Corpus :
# grobid_client --input pdfs/EIST_PDFS_TM --output xmls/EIST_PDFS_TM/ processFulltextDocument
# runtime: 383.979 seconds

# 3. RSOG Corpus :
# grobid_client --input pdfs/RSOG_PDFS_TM --output xmls/RSOG_PDFS_TM/ processFulltextDocument
# runtime: 698.183 seconds

# 4. SUS-SCI Corpus :
# grobid_client --input pdfs/SUS-SCI_PDFS_TM --output xmls/SUS-SCI_PDFS_TM/ processFulltextDocument
## Processing of SUS-SCI_PDFS_TM/s11625-021-00956-5.pdf failed with error 500 , [GENERAL] An exception occurred while running Grobid.
## Processing of SUS-SCI_PDFS_TM/s11625-021-00978-z.pdf failed with error 500 , [GENERAL] An exception occurred while running Grobid.
# runtime: 585.296 seconds