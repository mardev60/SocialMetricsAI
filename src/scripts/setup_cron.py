import os
import sys
import subprocess
from crontab import CronTab

def setup_cron_job():
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        retrain_script = os.path.join(current_dir, 'src', 'scripts', 'retrain.py')
        logs_dir = os.path.join(current_dir, 'logs')
        
        os.makedirs(logs_dir, exist_ok=True)
        
        os.chmod(retrain_script, 0o755)
        
        user = os.environ.get('USER', os.environ.get('USERNAME', 'root'))
        
        cron = CronTab(user=user)
        
        for job in cron.find_comment('SocialMetricsAI model retraining'):
            cron.remove(job)
            print("Ancien cronjob supprimé.")

        job = cron.new(command=f'{retrain_script} >> {logs_dir}/cron.log 2>&1')
        job.setall('0 3 * * 0')
        job.set_comment('SocialMetricsAI model retraining')
        
        cron.write()
        
        print(f"Cronjob configuré avec succès pour exécuter {retrain_script} chaque semaine.")
        print("Prochaines exécutions:")
        
        schedule = job.schedule(date_from=None)
        for i, execution in enumerate(schedule.get_next(3)):
            print(f"  {i+1}. {execution.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
    
    except Exception as e:
        print(f"Erreur lors de la configuration du cronjob: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_cron_job()
    sys.exit(0 if success else 1) 