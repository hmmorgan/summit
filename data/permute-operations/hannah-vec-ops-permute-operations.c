static char help[] = "Hannah: test vector operations on GPU\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            v,w,x,y;
  PetscInt       n=15;
  PetscScalar    val;
  PetscReal      norm1,norm2;
  PetscRandom    rctx;
  PetscLogStage  stage;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,PETSC_DECIDE,n);CHKERRQ(ierr);  
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  ierr = VecSetRandom(v,rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(w,rctx);CHKERRQ(ierr); 

  /* Hannah: create dummy vector to clear cache */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,10000000000);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);  
  ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);

  /* Hannah: first dot product */
  PetscBarrier(NULL);
  ierr = VecNorm(v,NORM_1,&norm1);

  /* Hannah: register a stage work on GPU */
  PetscLogStageRegister("Work on GPU", &stage);
  PetscLogStagePush(stage);
  ierr = VecNorm(w,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr); /* Hannah: clear cache */
  PetscBarrier(NULL);
  ierr = VecDot(w,v,&val);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_INFINITY,&norm1);CHKERRQ(ierr); /* Hannah: clear cache */
  PetscBarrier(NULL);
  ierr = VecAXPY(w,1.0,v);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm1);CHKERRQ(ierr); /* Hannah: clear cache */
  PetscBarrier(NULL);
  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm2);CHKERRQ(ierr); /* Hannah: clear cache */
  PetscBarrier(NULL);
  ierr = VecCopy(v,w);CHKERRQ(ierr);
  PetscLogStagePop();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Hannah: test completed successfully!\n");CHKERRQ(ierr);
  /* ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr); hannah: cheating */
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
