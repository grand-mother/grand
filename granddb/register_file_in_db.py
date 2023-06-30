# import sqlalchemy as sa
# from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.inspection import inspect
from sqlalchemy.dialects import postgresql

usr = "postgres"
psw = "password"
db = "granddb"
srv = 'lpndocker01.in2p3.fr'

server = SSHTunnelForwarder(
    ('lpnclaude.in2p3.fr', 22),
    ssh_username="fleg",
    ssh_pkey="/home/fleg/.ssh/id_rsa_decrypted",
    remote_bind_address=(srv, 5432)
)
server.start()
local_port = str(server.local_bind_port)
# create an engine
engine = create_engine('postgresql+psycopg2://' + usr + ':' + psw + '@' + '127.0.0.1:' + local_port + '/' + db)
Base = automap_base()
Base.prepare(engine, reflect=True)
session = Session(engine)
tables = {}
for table in engine.table_names():
    #if table not in ("file_location"):
        print(table)
        tables[table]=getattr(Base.classes, table)
        #klass = getattr(Base.classes, table)
        query = session.query(tables[table])
        #query = session.query(Base.classes.table)
        #query = session.query(Base.classes.repository)
        res = query.all()
        for resu in res:
            # print(resu.repository.name)
            print(resu.__dict__)

exit(0)
Repository = Base.classes.repository
Repository_access = Base.classes.repository_access
Protocol = Base.classes.protocol
Event = Base.classes.event
Shower_type = Base.classes.shower_type

# res = session.query(Repository).all()
# for resu in res:
#    print(resu.id_repository)

# res = session.query(Repository_access).join(Protocol).all()
# query = session.query(Repository_access, Repository).filter(Repository.id_repository == Repository_access.id_repository).filter(Repository_access.server_name == 'Gary')
query = session.query(Repository_access, Repository, Protocol).join(Repository).join(Protocol)

# res = session.query(Repository_access, Repository).filter(Repository.id_repository == Repository_access.id_repository).all()
res = query.all()
for resu in res:
    # print(resu.repository.name)
    print(resu.repository.__dict__)
    print(resu.repository_access.__dict__)

print(query.statement.compile(dialect=postgresql.dialect()))
print(len(res))

newproto = Protocol(id_protocol=None, name='http')
session.add(newproto)
session.commit()
session.refresh(newproto)
print("toto")
print(newproto.id_protocol)
# table = inspect(Repository)

# for column in table.c:
#    print(column.name)
#    print(column.type)
server.stop()
