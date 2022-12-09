import rootdblib as rdb



rfile = rdb.RootFile("../SIMUS/_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root")
#rfile = RootFile("SIMUS/Coarse2_xmax_add.root")
#rfile = RootFile("SIMUS/Coarse4.root")
print("list of trees:")
[print(treename) for treename in rfile.TreeList]
print()

for treename in rfile.TreeList:
    if treename in ["teventefield", "teventshowersimdata", "teventshowerzhaires", 'teventshower', 'teventvoltage']:
        print(treename)
        for event, run in rfile.TreeList[treename].get_list_of_events():
            rfile.TreeList[treename].get_event(event,run)
            #print("EVENT : " + str(rfile.TreeList[treename].event_number) + " RUN "+ str(rfile.TreeList[treename].run_number))
            for param, field in getattr(rfile, treename + "ToDB").items():
                if param == "table":
                    print("values will go to table " + field)
                    table = field
                else:
                    print("param " + param + " with value " + str(
                        getattr(rfile.TreeList[treename], param)) + " goes to field " + field + " in table " + table)

    if treename in ["trun", "trunefieldsimdata"]:
        print(treename)
        for run in rfile.TreeList[treename].get_list_of_runs():
            print("RUN IS:" + str(run))
            rfile.TreeList[treename].get_run(run)
            for param, field in getattr(rfile, treename + "ToDB").items():
                if param == "table":
                    print("values will go to table " + field)
                    table = field
                else:
                    print("param " + param + " with value " + str(
                        getattr(rfile.TreeList[treename], param)) + " goes to field " + field + " in table " + table)


#print(rfile.TreeList['teventvoltage'])
#print(rfile.TreeList['trun'])

