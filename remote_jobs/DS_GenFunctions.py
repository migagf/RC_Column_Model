def DS_GetDir(cur_dir, ag):

#cur_dir = os.getcwd()
    if ('jupyter/MyData' in cur_dir ):
        cur_dir = cur_dir.split('MyData').pop() 
        storage_id = 'designsafe.storage.default'
        input_dir = ag.profiles.get()['username']+ cur_dir
        input_uri = 'agave://{}/{}'.format(storage_id,input_dir)
        input_uri = input_uri.replace(" ","%20")
    elif('jupyter/mydata' in cur_dir ):
        cur_dir = cur_dir.split('mydata').pop()
        storage_id = 'designsafe.storage.default'
        input_dir = ag.profiles.get()['username']+ cur_dir
        input_uri = 'agave://{}/{}'.format(storage_id,input_dir)
        input_uri = input_uri.replace(" ","%20")
    elif('jupyter/MyProjects' in cur_dir):
        cur_dir = cur_dir.split('MyProjects/').pop()
        PRJ = cur_dir.split('/')[0]
        qq = {"value.projectId": str(PRJ)}
        cur_dir = cur_dir.split(PRJ).pop()
        project_uuid = ag.meta.listMetadata(q=str(qq))[0]["uuid"]
        input_dir = cur_dir
        input_uri = 'agave://project-{}{}'.format(project_uuid,cur_dir)
        input_uri = input_uri.replace(" ","%20")
    elif('jupyter/projects' in cur_dir):
        cur_dir = cur_dir.split('projects/').pop()
        PRJ = cur_dir.split('/')[0]
        qq = {"value.projectId": str(PRJ)}
        cur_dir = cur_dir.split(PRJ).pop()
        project_uuid = ag.meta.listMetadata(q=str(qq))[0]["uuid"]
        input_dir = cur_dir
        input_uri = 'agave://project-{}{}'.format(project_uuid,cur_dir)
        input_uri = input_uri.replace(" ","%20")    
    elif('jupyter/CommunityData' in cur_dir):
        cur_dir = cur_dir.split('jupyter/CommunityData').pop() 
        input_dir = cur_dir
        input_uri = 'agave://designsafe.storage.community/{}'.format(input_dir)
        input_uri = input_uri.replace(" ","%20")    
        
    return input_uri

def DS_GetStatus(ag, mjobId, tlapse = 15):
    import time
    # Get job status
    # ag = Agave job
    status = ag.jobs.getStatus(jobId=mjobId)["status"]
    previous = ""
    while True:
        if status in ["FINISHED","FAILED","STOPPED"]:
            break
        status = ag.jobs.getStatus(jobId=mjobId)["status"]
        if status == previous:
            continue
        else :
            previous = status
        print(f"\tStatus: {status}")
        time.sleep(tlapse)    
    return status 

def DS_GetRuntime(ag, jobid):
    print("\nRuntime Summary")
    print("---------------")
    hist = ag.jobs.getHistory(jobId=jobid)
    print("TOTAL   time:", hist[-1]["created"] - hist[0]["created"])
    
    for i in range(len(hist)):
        if hist[i]["status"] == 'RUNNING' :
            print("RUNNING time:", hist[i+1]["created"] - hist[i]["created"])
        if hist[i]["status"] == 'QUEUED' :
            print("QUEUED  time:", hist[i+1]["created"] - hist[i]["created"])