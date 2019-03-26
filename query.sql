Query ByPrice
SELECT Students.ID, 
Students.Name, 
Friends.Friend_id, 
Packages.Salary AS Salary_Friend, 
Packages.Salary AS Student_Salary
FROM Students
INNER JOIN FRIENDS
ON Students.ID = Friends.ID
INNER JOIN Packages
ON Friends.Friend_id = Packages.ID
INNER JOIN Packages 
ON Students.ID = Packages.ID
ORDER BY Salary_Friend DESC;

-----------------------------------------

SELECT t1.s_id 
FROM (SELECT Students.ID AS s_id, 
Students.Name AS s_name, 
Friends.Friend_id AS f_id, 
FROM Students
INNER JOIN FRIENDS
ON Students.ID = Friends.ID) t1
INNER JOIN Students t2
ON t1.f_id=t2.ID

-----------EL bueno!!!-----------------

SELECT s_name FROM (SELECT * FROM (SELECT t5.*, t6.Salary AS f_salary FROM 
               (SELECT t3.*, t4.Salary AS s_salary FROM (
Select t1.s_id, t1.s_name, t1.f_id, t2.name AS f_name
FROM (SELECT Students.ID AS s_id, 
Students.Name AS s_name, 
Friends.Friend_id AS f_id
FROM Students
INNER JOIN FRIENDS
ON Students.ID = Friends.ID) t1
INNER JOIN Students t2
ON t1.f_id=t2.ID) t3
INNER JOIN Packages t4
ON t3.s_id=t4.ID) t5
INNER JOIN Packages t6
ON t5.f_id=t6.ID) t7
WHERE f_salary > s_salary
ORDER BY f_salary);